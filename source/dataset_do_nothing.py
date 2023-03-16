import os
import numpy as np
#import rasterio
import re
import random
#import tifffile as tif


import torch
from torch.utils.data import Dataset as BaseDataset
from . import transforms as aug
from torchvision import transforms as T

from scipy.fftpack import dct, idct
from skimage.exposure import match_histograms
from PIL import Image


"""def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)"""

def make_masks(r_list):
    """
    return the binary matrix to extract band-pass, size: (block_size, block_size, len(r_list))
    """
    
    block_size = 512
    masks = []
    before_mask = np.zeros((block_size, block_size))
    for m in range( len(r_list) ):
        matrix = np.zeros((block_size, block_size))
        for i in range(block_size):
            for j in range(block_size):
                d = i**2 + j**2
                if d <=r_list[m]**2:
                    #print(m, i, j)
                    matrix[i, j] = 1
                else:
                    #print(m, i,j , "break")
                    break
            if (d>r_list[m]**2) and (j==0):
                #print(m, i,j , "Break")
                break
        masks.append(matrix - before_mask)
        before_mask = matrix
    masks.append(np.ones((block_size, block_size) ) - before_mask)
    return masks

def dct2(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def exact_feature_distribution_matching(content, style):
    assert (content.size() == style.size()) ## content and style features should share the same shape
    #B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
    W, H = content.size(0), content.size(1)
    _, index_content = torch.sort(content.view(-1))  ## sort content feature
    value_style, _ = torch.sort(style.view(-1))      ## sort style feature
    inverse_index = index_content.argsort(-1)
    transferred_content = content.view(-1) + value_style.gather(-1, inverse_index) - content.view(-1).detach()
    return transferred_content.view(W, H).numpy()


class Dataset(BaseDataset):
    def __init__(self, img_list, ano_list, ref_list = None, classes=None, \
        size=512, train=False, n_input_channels = None, randomize_prob = None, c_fc_values = None):
        
        self.img_list = img_list
        self.ano_list = ano_list
        self.ref_list = ref_list 
        self.concatenate_list = self.img_list + self.ref_list
        
        self.augm = aug.train_augm if train else aug.test_augm
        self.size = size
        self.train = train
        self.USE_RAW_RGB = True if n_input_channels == 3 else False
        self.randomize_prob = randomize_prob
        
        self.t = T.Compose([T.ToTensor()])
        self.mask2tensor = aug.mask2tensor(classes = classes)
        
        
        #----------
        # for randomize FCs' parameters
        r_list = [8,  16, 32, 64, 96, 128,  256, 512]
        self.masks = make_masks(r_list)
        if self.train:
            self.c_fc_values = c_fc_values
        #----------
        
        self.randcrop_for_ref = T.RandomCrop(self.size, pad_if_needed=True)

    def random_crop(self, img, ano, ref):
        """
        random crop the img and ano in the same way
        """
        if (img.size[0]>self.size) and (img.size[1]>self.size):
            #Randomly decide the position to crop image
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.size, self.size))
        else:
            #For the case where the img size is smaller than self.size (512)
            i = random.randint(img.size[0] - self.size, 0)
            j = random.randint(img.size[1] - self.size, 0)
            h, w = self.size, self.size

        croped_img = T.functional.crop(img, i, j, h, w)
        croped_ano = T.functional.crop(ano, i, j, h, w) 

        if ref is None:
            return np.array(croped_img), np.array(croped_ano), None
        
        #For the trainging phase, ref is also randomly cropped 
        croped_ref = self.randcrop_for_ref(ref)
        return np.array(croped_img), np.array(croped_ano), np.array(croped_ref)
    
    def randomize_fcs(self, img, ref, randomize = False):
        """
        For test phase or not randimized train img, img is not randomized (randomize = False), and just separate FCs
        For randomize train img, img is randomized by Histgram Match (randomize = True)
        return Tensor(512, 512, len(masks)*3)
        """
    
        img = img.astype(np.float32) / 255.0
        imF = dct2(img)
        
        if randomize:
            ref = ref.astype(np.float32) / 255.0
            reF = dct2(ref)
        
        dct_img = np.zeros((self.size, self.size, 3*len(self.masks)), dtype = "float")

        for rgb in range(3):
            for i in range(len(self.masks)):
                idx_fc = len(self.masks)*rgb+i 

                if randomize and idx_fc in self.c_fc_values.random_fcs:
                    # to randomize FCs for train phase 
                    #dct_img[:, :, idx_fc] = idct2(match_histograms(imF[:, :, rgb]*self.masks[i], reF[:,  :, rgb] * self.masks[i]))
                    dct_img[:, :, idx_fc] = idct2(exact_feature_distribution_matching(torch.Tensor(imF[:, :, rgb]*self.masks[i]), torch.Tensor(reF[:,  :, rgb] * self.masks[i])))
                else:
                    dct_img[:, :, idx_fc] = idct2(imF[:, :, rgb]*self.masks[i])

        return dct_img
    
    def __getitem__(self, idx):
        domain, name  = re.findall("/integrated/(.*)/images/(.*).tif", self.img_list[idx] )[0]
        #print(domain, name)
        
        img = Image.open(self.img_list[idx])
        msk = Image.open(self.ano_list[idx])
        ref = None

        if self.train:
            if not self.USE_RAW_RGB:
                randomize = np.random.rand()<self.randomize_prob
                #print(randomize)
                if randomize:
                    ref_idx = np.random.choice(len(self.concatenate_list))
                    ref = Image.open(self.concatenate_list[ref_idx])
                
                img, msk, ref = self.random_crop(img, msk, ref)
                dct_img = self.randomize_fcs(img, ref, randomize)

            else:
                #use RGB image, dct_img is just RGB img
                dct_img, msk, ref = self.random_crop(img, msk, ref)
        
        else:
            #For valid phase and test phase, img is randomily cropped to be 512*512 size
            if not self.USE_RAW_RGB:
                img, msk, ref = self.random_crop(img, msk, ref)
                dct_img = self.randomize_fcs(img, None, randomize=False)
            else:
                dct_img, msk, ref = self.random_crop(img, msk, ref)
            
        data = self.augm({"image": dct_img, "mask": msk}, size = self.size)
        data["image"] = self.t(data["image"])
        data["mask"] = self.mask2tensor(data["mask"])
        if not self.train:
            #test phase
            return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx], "shape": msk.shape, "domain":domain}
        
        return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx]}
    
    def __len__(self):
        return len(self.img_list)
    

    

