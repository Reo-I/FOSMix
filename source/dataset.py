import os
import numpy as np
import re
import random
import rasterio
from PIL import Image
import albumentations as A


import torch
from torch.utils.data import Dataset as BaseDataset
from . import transforms as aug
from torchvision import transforms as T



def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)

def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)

class Dataset(BaseDataset):
    def __init__(self, img_list, ano_list, ref_list = None, classes=None, \
        size=512, train=False, final = False, randomize = True, args = None, \
        p_down_scale = None):
        
        self.img_list = img_list
        self.ano_list = ano_list
        self.ref_list = ref_list
        self.concatenate_list = self.img_list + self.ref_list
        self.args = args
        self.color_aug = aug.color_augm
        if train:
            self.augm = aug.train_augm
        elif final:
            self.augm = aug.final_augm
        else:
            self.augm = aug.test_augm
        #self.augm = aug.train_augm if train else aug.test_augm
        self.size = size
        self.train = train
        self.final = final
        self.randomize = randomize
        self.p_down_scale = p_down_scale
        
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        #self.t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        self.t = T.Compose([T.ToTensor()])
        self.mask2tensor = aug.mask2tensor(classes = classes)

        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

        self.pad = A.Compose([A.PadIfNeeded(min_height=self.size, min_width=self.size)])
        self.randcrop_for_ref = T.RandomCrop(self.size, pad_if_needed=True, padding_mode = "symmetric")

        if self.args.prob_imagenet_ref>0:
            self.imgnet_path = self.args.imgnet_path
            self.imgnet_list = [f for f in os.listdir(self.imgnet_path) \
                if os.path.isdir(os.path.join(self.imgnet_path, f))]

    def random_crop(self, img, ano, ref):
        """
        random crop the img and ano in the same way
        """
        if (img.size[0]>self.size) and (img.size[1]>self.size):
            #Randomly decide the position to crop image
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.size, self.size))
            img = np.array(T.functional.crop(img, i, j, h, w))
            ano = np.array(T.functional.crop(ano, i, j, h, w))
        elif (img.size[0]<self.size) or (img.size[1]<self.size):
            #For the case where the img size is smaller than self.size (512)
            i = random.randint(img.size[0] - self.size, 0)
            j = random.randint(img.size[1] - self.size, 0)
            h, w = self.size, self.size

            croped_data = self.pad(image=np.array(img), mask = np.array(ano))
            img = croped_data["image"]
            ano = croped_data["mask"]

        if ref is None:
            return img, ano, None
        
        #For the trainging phase, ref is also randomly cropped 
        ref = self.randcrop_for_ref(ref)
        return img, ano, np.array(ref)
    

    def __getitem__(self, idx):
        if self.args.dataset == "OEM":
            domain, name  = re.findall("/openearthmap/(.*)/images/(.*).tif", self.img_list[idx] )[0]
            img = Image.open(self.img_list[idx])
            msk = Image.open(self.ano_list[idx])
        elif self.args.dataset == "FLAIR":
            domain, name  = re.findall("/flair/(.*)/.*/img/(.*).tif", self.img_list[idx] )[0]
            img = self.load_multiband(self.img_list[idx])[:, :, :3]
            msk = self.load_grayscale(self.ano_list[idx])
            msk[msk>12] = 0
        ref = None

        if self.train and self.randomize:
            if self.args.server == "wisteria" and random.random()<self.args.prob_imagenet_ref:
                #ref from ImageNet
                while True:
                    sampled_c = random.choice(self.imgnet_list) + "/"
                    sampled_n = random.choice(os.listdir(self.imgnet_path + sampled_c))
                    ref = Image.open(self.imgnet_path + sampled_c + sampled_n)
                    if ref.mode == "RGB":
                        break
                ref = T.Resize(self.size)(ref)

            elif not self.args.use_only_ref:
                #ref from reference domains or source domains 
                ref_idx = np.random.choice(len(self.concatenate_list), 
                    p = [1/2*(1/len(self.img_list))]*len(self.img_list) \
                        + [1/2*(1/len(self.ref_list))]*len(self.ref_list)
                )
                ref = Image.open(self.concatenate_list[ref_idx]) if self.args.dataset == "OEM" \
                else self.load_multiband(self.concatenate_list[idx])[:, :, :3]
            else:
                #ref from reference domains
                ref_idx = np.random.choice(len(self.ref_list))
                ref = Image.open(self.ref_list[ref_idx]) if self.args.dataset == "OEM" \
                else self.load_multiband(self.ref_list[idx])[:, :, :3]
            
            if self.args.dataset == "OEM":
                img, msk, ref = self.random_crop(img, msk, ref)
                if self.args.pb_set == "resolution":
                    ref = A.Downscale(scale_min=0.1, scale_max=0.8, p=0.8)(image=ref)['image']

        elif (not self.final) or (self.args.test_crop):
            #For Validation and Test phases, and for no_randomize ver.
            if self.args.dataset == "OEM":
                img, msk, ref = self.random_crop(img, msk, ref)
        
        if self.train and (self.args.dataset == "OEM" and self.args.pb_set == "resolution"):
            img = A.Downscale(scale_min=0.1, scale_max=0.8, p=0.8)(image=img)['image']
        

        data = self.augm({"image": img, "mask": msk}, size = self.size , args = self.args, p_down_scale= self.p_down_scale)
        if self.final:
            data["image"] = torch.cat([self.t(x).unsqueeze(0) for x in data["image"]], dim=0).float()
            return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx], "shape": np.array(msk).shape, "domain":domain}
        elif self.randomize:
            data["color_x"] = self.color_aug({"image":data["image"]}, args = self.args)["image"]
            data["color_x"] = self.t(data["color_x"])

        data["image"] = self.t(data["image"])
        data["mask"] = self.mask2tensor(data["mask"])
        
        if (not self.train) or (not self.randomize):
            #test phase or no_randomized training
            return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx], "shape": msk.shape, "domain":domain}
        
        data["ref"] = self.t(ref) 
        return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx], "ref":data["ref"], "color_x":data["color_x"]}
    
    def __len__(self):
        return len(self.img_list)
