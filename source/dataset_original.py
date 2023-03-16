import os
import numpy as np
import rasterio
from scipy.fftpack import idct, dct

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
    def __init__(self, img_list, ano_list, classes=None, size=256, train=False, test = False, for_show = False, opt=None):
        self.img_list = img_list
        self.ano_list = ano_list
        self.opt = opt

        if train:
            self.augm = aug.train_augm
        elif not test:
            self.augm = aug.valid_augm
        else:
            self.augm = aug.test_augm
        #self.augm = aug.train_augm if train else aug.valid_augm
        self.size = size
        self.train = train
        self.test = test
        self.for_show = for_show
        #self.to_tensor = T.ToTensor(classes=classes)
        self.mask2tensor = aug.mask2tensor(classes = classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        

        self.t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])


    def __getitem__(self, idx):
        img = self.load_multiband(self.img_list[idx])
        msk = self.load_grayscale(self.ano_list[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size, opt = self.opt)

        elif not self.test:
            data = self.augm({"image": img, "mask": msk}, 512)

        elif self.test:
            data, shape = self.augm({"image": img, "mask": msk})
            data["image"] = torch.cat([self.t(x).unsqueeze(0) for x in data["image"]], dim=0).float()
            return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx], "shape":shape, "raw":img}

        if not self.for_show:
            data["image"] = self.t(data["image"])
        data["mask"] = self.mask2tensor(data["mask"])
        #data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.img_list[idx]}

    def __len__(self):
        return len(self.img_list)
