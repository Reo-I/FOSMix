import warnings
import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF

import cv2
import math

# reference: https://albumentations.ai/
warnings.simplefilter("ignore")

class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) for v in self.classes]
        msk = np.stack(msks, axis=-1).astype(np.float32)
        background = 1 - msk.sum(axis=-1, keepdims=True)
        sample["mask"] = TF.to_tensor(np.concatenate((background, msk), axis=-1))

        for key in [k for k in sample.keys() if k != "mask"]:
            sample[key] = TF.to_tensor(sample[key].astype(np.float32) / 255.0)
        return sample

class mask2tensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, mask):
        msks = [(mask == v) for v in self.classes]
        msk = np.stack(msks, axis=-1).astype(np.float32)
        background = 1 - msk.sum(axis=-1, keepdims=True)
        mask = TF.to_tensor(np.concatenate((background, msk), axis=-1))
        return mask

def train_augm(sample, size=512, args = None, p_down_scale = None):
    augms = [
        A.Rotate(p = 0.75),
        #A.RandomCrop(size, size, p=1.0),
        #default: p = 0.75
        A.Flip(p=0.75), 
        A.GaussNoise(p=0.1),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def color_augm(sample, args = None):
    augms = [  
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1),
                A.HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1
                ),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
                A.ToGray(p=1),
                A.ToSepia(p=1),
            ],
            p=args.aug_color,
        ),
    ]
    return A.Compose(augms)(image=sample["image"])


def test_augm(sample, size = 512, args = None, p_down_scale = None):
    #do nothing
    if args.pb_set == "resolution":
        sample["image"] =A.Downscale(scale_min=args.down_scale, scale_max=args.down_scale, \
            p=p_down_scale)(image=sample["image"])['image']
    return sample


def final_augm(sample, size = 512, args = None, p_down_scale = None):
    img = np.array(sample["image"])
    if args.pb_set == "resolution":
        img = A.Downscale(scale_min=args.down_scale, scale_max=args.down_scale, p=1)(image=img)['image']
    h, w = img.shape[:2]
    power = math.ceil(np.log2(h) / np.log2(2))
    shape = (2 ** power, 2 ** power)
    img = cv2.resize(img, shape)

    imgs = []
    imgs.append(img.copy())
    imgs.append(img[:, ::-1, :].copy())
    imgs.append(img[::-1, :, :].copy())
    imgs.append(img[::-1, ::-1, :].copy())
    
    return {"image":imgs, "mask":np.array(sample["mask"])}