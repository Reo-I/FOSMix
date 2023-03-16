
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import albumentations as A

phase = ["train", "val", "test"]
img_file = [[] for _ in range(len(phase))]
ano_file = [[] for _ in range(len(phase))]

for p in range(len(phase)):
    with open("./split_file/"+phase[p]+"_path_fns.txt", "r") as f:
            img_file[p] =  f.read().split("\n")[:-1]
    ano_file[p] = list(map(lambda x: x.replace('images', 'labels'), img_file[p]))


import rasterio
from scipy.fftpack import idct, dct

def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)

def valid_augm(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])

def make_masks(block_size, r_list = [0, 1, 2, 3, 4, 5, 6, 8]):
    masks = []
    before_mask = np.zeros((block_size, block_size))
    for m in range( len(r_list) ):
        matrix = np.zeros((block_size, block_size))
        for i in range(block_size):
            for j in range(block_size):
                d = i**2 + j**2
                if d <=r_list[m]**2:
                    matrix[i, j] = 1
                else:
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


r_list = [0, 1, 2, 3, 4, 5, 6, 8]
mu = np.zeros(3*(len(r_list)+1))
sigma = np.zeros(3*(len(r_list)+1))

block_size = 8

#make masks
masks = make_masks(block_size, r_list)
num_masks = len(masks)


files = img_file[0]+img_file[1]+img_file[2]
mask = np.zeros((256, 256))
count = 0
for idx in tqdm(range(len(files))):
    img = load_multiband(files[idx])
    data = valid_augm({"image": img, "mask": mask}, 256)
    
    
    imsize = data["image"].shape
    mu_each = np.zeros(3*(len(r_list)+1))
    sigma_each = np.zeros(3*(len(r_list)+1))
    #masked_img = np.zeros((imsize[0], imsize[1], imsize[2]*(len(r_list)+1)))
    
    # Do 8x8 DCT on image (in-place)
    for rgb in range(3):
        for i in np.r_[:imsize[0]:block_size]:
            for j in np.r_[:imsize[1]:block_size]:
                fre_img = dct2(data["image"][i:(i+block_size),j:(j+block_size), rgb])
                for r in range(num_masks):
                    filtered_img = idct2(masks[r] * fre_img)
                    mu_each[ rgb*num_masks+r ] += np.sum(filtered_img)
                    sigma_each[ rgb*num_masks+r ] += np.sum(filtered_img**2)
    mu += mu_each / (256*256)
    sigma += sigma_each / (256*256)
mu = mu/len(files)
sigma1= sigma/len(files) - mu**2

np.save('mu.npy', mu)
np.save('sigma.npy', sigma1)