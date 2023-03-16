import re
import glob
import random
import os
from collections import defaultdict


def extract_fns(path, f_path = None):
    img_path = glob.glob(path)
    phase = ["train", "ref", "test", "val"]
    fns = [[] for _ in range(len(phase))]

    for i in range(len(phase)):
        with open("./"+f_path+phase[i]+"_fns.txt", "r") as f:
            fns[i] = f.read().split("\n")[:-1]

    path_fns = [[] for _ in range(len(phase))]

    for i in img_path:
        img_name = re.findall('/.*/openearthmap/.*/images/(.*.tif)', i)[0]
        if img_name in fns[0]:
            path_fns[0].append(i)
        elif img_name in fns[1]:
            path_fns[1].append(i)
        elif img_name in fns[2]:
            path_fns[2].append(i)
        elif img_name in fns[3]:
            path_fns[3].append(i)


    for i in range(len(phase)):
        f = open('./'+f_path+phase[i]+'_path_fns.txt', 'w')
        for x in path_fns[i]:
            f.write(str(x) + "\n")
        f.close()

class ExtractData:
    """
    Extract data and split train, ref, test, and val
    """
    
    def __init__(self, dataset, path) -> None:
        self.phase = ["train", "ref", "test", "val"]
        self.img_file = [[] for _ in range(len(self.phase))]
        self.ano_file = [[] for _ in range(len(self.phase))]
        self.phase_domains = [[] for _ in range(len(self.phase))]
        self.dataset = dataset
        self.path = path

    def extract_fns_flair(self):
        #FLAIR dataset
        with open("./split_file_FLAIR/val.txt") as f:
            self.phase_domains[3] = f.read().splitlines()
            for d in self.phase_domains[3]:
                self.img_file[3].extend(glob.glob(self.path + d + "/*/img/*.tif"))
        
        all_domains = sorted(set([d_name for d_name in os.listdir(self.path)\
                                   if re.fullmatch(r'D0\d{2}_20\d{2}', d_name)]) \
                             - set(self.phase_domains[3]))
        for d in all_domains:
            d_num = int(d[2:4])
            if d_num %3 == 0:
                #test, multiples of 3
                self.phase_domains[2].append(d)
                self.img_file[2].extend(glob.glob(self.path + d + "/*/img/*.tif"))
            elif d_num %4 ==0 or d_num%5 == 0:
                #train, multiples of 4 and 5
                self.phase_domains[0].append(d)
                self.img_file[0].extend(glob.glob(self.path + d + "/*/img/*.tif"))
            else:
                #ref, others
                self.phase_domains[1].append(d)
                self.img_file[1].extend(glob.glob(self.path + d + "/*/img/*.tif"))
        for p in range(len(self.phase)):
            self.ano_file[p] = list(map(lambda x: x.replace('img', 'msk').replace('IMG', 'MSK'), \
                                        self.img_file[p]))
        return self.img_file, self.ano_file



    
def extract_fns_flair(path, phase):
    img_file = [[] for _ in range(len(phase))]
    ano_file = [[] for _ in range(len(phase))]
    phase_domains = [[] for _ in range(len(phase))]

    with open("./split_file_FLAIR/val.txt") as f:
        phase_domains[3] = f.read().splitlines()
        for d in phase_domains[3]:
            img_file[3].extend(glob.glob(path + d + "/*/img/*.tif"))

    all_domains = sorted(set([d_name for d_name in os.listdir(path) if re.fullmatch(r'D0\d{2}_20\d{2}', d_name)])  - set(val_d))
    for d in all_domains:
        d_num = int(d[2:4])
        if d_num %3 == 0:
            #test, multiples of 3
            phase_domains[2].append(d)
            img_file[2].extend(glob.glob(path + d + "/*/img/*.tif"))
        elif d_num %4 ==0 or d_num%5 == 0:
            #train, multiples of 4 and 5
            phase_domains[0].append(d)
            img_file[0].extend(glob.glob(path + d + "/*/img/*.tif"))
        else:
            #ref, others
            phase_domains[1].append(d)
            img_file[1].extend(glob.glob(path + d + "/*/img/*.tif"))
    for p in range(len(phase)):
        ano_file[p] = list(map(lambda x: x.replace('img', 'msk').replace('IMG', 'MSK'), img_file[p]))
    return img_file, ano_file
