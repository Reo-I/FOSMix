import random
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import OrderedDict

from . import dct_torch
from . import networks


class GenerateMask(nn.Module):
    def __init__(self, nz):
        super(GenerateMask, self).__init__()
        self.nz = nz
        self.ngf = 64
        self.nc = 3

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 8, 4, 2, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d( self.ngf, self.nc, 8, 4, 2, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 512 x 512
        )

    def forward(self, input):
        return self.main(input)

class FCsMix(nn.Module):
    """
    FCsMix (= Randomize FCs for generalization)
    """
    def __init__(self, args = False):
        """
        Args:
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self._activated = True
        #---------
        self.args = args
        #----------

        if self.args.optimize and (not self.args.MFI):
            self.maskG = GenerateMask(nz = 100)
        elif self.args.optimize and self.args.MFI:
            self.maskG = networks.load_MFI_network(args=args)
        elif self.args.randomize and (not self.args.optimize):
            pass
        
        if self.args.optimize and (not self.args.MFI):
            self.maskG = GenerateMask(nz = 100)
        elif self.args.optimize and self.args.MFI:
            self.maskG = networks.load_MFI_network(args=args)
        elif self.args.randomize and (not self.args.optimize):
            pass
        
        if self.args.pb_set == "resolution":
            row = np.array(range(512))
            self.xx, self.yy = np.meshgrid(row,row)
            
            
    def forward(self, x, ref=None):
        imF = dct_torch.dct_2d(x, norm = "ortho")
        reF = dct_torch.dct_2d(ref, norm = "ortho")

        if not self.training or not self._activated:
            return x

        B, C, W, H = imF.size(0), imF.size(1), imF.size(2), imF.size(3)
        mk_opt, mk_full = None, None
        if self.args.optimize:
            #Mask the mask from Generator
            if not self.args.MFI:
                noise = torch.randn(1, 100, 1, 1, device=x.device)
                mk_opt = self.maskG(noise)
                mk_opt = mk_opt.reshape(C, -1)
            elif self.args.MFI:
                if self.args.G_use_ref == 0:
                    mk_opt = self.maskG(x)
                else:
                    mk_opt = self.maskG(torch.cat([x, ref], dim=1))
                mk_opt = mk_opt.reshape(B, C, -1)
            
        if self.args.fullmask:
            #Make mask with all zeros
            mk_full = torch.zeros((C, W, H), device=x.device)
            mk_full = mk_full.reshape(C, -1)
        elif self.args.halfmask:
            #Make mask with 0 or 1 randomly
            mk_full = torch.randint(low=0, high=2, size=(C, W, H), device=x.device, dtype =torch.float)
            mk_full = mk_full.reshape(C, -1)

        """
        _, index_content = torch.sort(imF.reshape(B,C,-1))  ## sort content feature
        value_style, _ = torch.sort(reF.contiguous().reshape(B,C,-1))      ## sort style feature
        inverse_index = index_content.argsort(-1)
        transferred_content = imF.reshape(B,C,-1) + value_style.gather(-1, inverse_index)*(1 - mk) - imF.reshape(B,C,-1).detach()*(1 - mk)

        if not self.args.MFI:
            return dct_torch.idct_2d(transferred_content.view(B, C, W, H), norm = "ortho"), mk.view(-1, W, H)
        return dct_torch.idct_2d(transferred_content.view(B, C, W, H), norm = "ortho"), mk.view(B, C, W, H)
        """

        """
        if self.args.pb_set == "resolution":
            start_r_list = [32, 64, 128, 256]
            s_i = random.choice(range(len(start_r_list)))
            start_r = start_r_list[s_i]
            end_r = random.choice(start_r_list[s_i+1:]+[512])

            bpf = torch.ones((3, 512, 512), device=reF.device)
            print(start_r, end_r)
            bpf[:, (np.sqrt(self.xx**2 + self.yy**2)>start_r) &(np.sqrt(self.xx**2 + self.yy**2)<=end_r)] = 0
            reF = reF * bpf
        """

        opt_mixed_img, full_mixed_img = EFDM(imF, reF, mk_opt=mk_opt, mk_full=mk_full, shape=imF.size())
        if (not self.args.optimize) and (self.args.fullmask or self.args.halfmask):
            return None, full_mixed_img, None

        return opt_mixed_img, full_mixed_img, mk_opt.view(-1, W, H)
    
    def load_parameters(self, netG_path, device):
        state_dict = torch.load(netG_path,map_location=device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if device == "cpu" and 'module' in k:
                k = k.replace('module.', '')
            if 'maskG' in k:
                k = k.replace('maskG.', '')
            new_state_dict[k] = v
        self.maskG.load_state_dict(new_state_dict)
            
        self.maskG.eval()

        
def EFDM(imF, reF, mk_opt, mk_full, shape):
    B, C, W, H = shape
    _, index_content = torch.sort(imF.reshape(B,C,-1))  ## sort content feature
    value_style, _ = torch.sort(reF.contiguous().reshape(B,C,-1))      ## sort style feature
    inverse_index = index_content.argsort(-1)

    opt_mixed, full_mixed = None, None
    if mk_opt is not None:
        transferred_content = imF.reshape(B,C,-1) + value_style.gather(-1, inverse_index)*(1 - mk_opt) - imF.reshape(B,C,-1).detach()*(1 - mk_opt)
        opt_mixed = dct_torch.idct_2d(transferred_content.view(B, C, W, H), norm = "ortho")
    if mk_full is not None:
        transferred_content = imF.reshape(B,C,-1) + value_style.gather(-1, inverse_index)*(1 - mk_full) - imF.reshape(B,C,-1).detach()*(1 - mk_full)
        full_mixed = dct_torch.idct_2d(transferred_content.view(B, C, W, H), norm = "ortho")

    return opt_mixed, full_mixed


    