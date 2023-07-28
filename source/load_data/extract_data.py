import re
import glob
import random
import os

class ExtractData:
    """
    Extract data and split train, ref, test, and val

    Attributes:
        dataset (str): The dataset to extract data from.
        path (str): path to the dataset
    """
    
    def __init__(self, dataset:str, path:str) -> None:
        self.phase = ["train", "ref", "test", "val"]
        self.img_file = [[] for _ in range(len(self.phase))]
        self.ano_file = [[] for _ in range(len(self.phase))]
        self.phase_domains = [[] for _ in range(len(self.phase))]
        self.dataset = dataset
        self.path = path

    def extract_fns_oem(self, pb_set:str):
        # OEM dataset
        pb_name = 'split_file/' if pb_set == "region" else 'split_file_for_resolution/'
        fns = [[] for _ in range(len(self.phase))]
            
        for idx, phase in enumerate(self.phase):
            with open(f"./{pb_name}{phase}_fns.txt", "r", encoding='utf-8') as f:
                fns[idx] = f.read().split("\n")[:-1]
            self.img_file[idx] = list(map(
                lambda x: self.path+re.findall("(.*)_\d*.tif", x)[0]+ "/images/"+x, fns[idx]
            ))
            self.ano_file[idx] = list(map(
                lambda x: x.replace('images', 'labels'), self.img_file[idx]\
            ))
        return self.img_file, self.ano_file
    
    def extract_fns_flair(self):
        # FLAIR dataset
        with open("./split_file_FLAIR/val.txt", encoding='utf-8') as f:
            self.phase_domains[3] = f.read().splitlines()
            for d in self.phase_domains[3]:
                self.img_file[3].extend(glob.glob(self.path + d + "/*/img/*.tif"))
        
        all_domains = sorted(
            set([d_name for d_name in os.listdir(self.path) if re.fullmatch(r'D0\d{2}_20\d{2}', d_name)])
            - set(self.phase_domains[3])
        )
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
    
