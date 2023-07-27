import argparse
from ast import parse

class BaseOptions():
    def __init__(self) -> None:
        pass

    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="OEM", \
            help="OEM: OpenEarthMap or FLAIR: https://codalab.lisn.upsaclay.fr/competitions/8769#participate ")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=12)
        parser.add_argument("--n_epochs", type=int, default=100)
        parser.add_argument('--ver', type=str, default='1', help='version of experiment')
        parser.add_argument('--loss_type', type=str, default='jaccard', help='loss function, jaccard/bce')
        parser.add_argument("--n_class", type = int,  default=8)
        parser.add_argument("--final", type = int,  default=0)
        parser.add_argument("--use_only_ref", type=int, default=0)
        
        parser.add_argument("--randomize", type = int,  default=1, \
            help="conduct the experiments with/without randomizing FCs")
        parser.add_argument("--optimize", type = int,  default=1, \
            help = "whether mask M is optimized or not")
        parser.add_argument("--LAMBDA1", type = float,  default=5., \
            help = "the weight of LASSO for masks")
        parser.add_argument("--LAMBDA2", type = float,  default=0.1, \
            help = "the weight of variance for masks")
        parser.add_argument("--aug_color", type = float,  default=0.5, \
            help = "the probability to augment images with color")

        parser.add_argument("--MFI", type=int, default=0, \
            help = "whether make the Masks From input Images(1) or noise (0)")
        parser.add_argument("--fullmask", type=int, default=0, \
            help = "Full Mask is that all elements of mask are 0, which means the output is the most stylized image")
        parser.add_argument("--halfmask", type=int, default=0, \
            help = "Half Mask is to randomize half of FCs")
        parser.add_argument("--COF", type = int, default =0,\
            help = "add Consistency loss between opt_mixed_img and full_mixed_img")
        parser.add_argument("--COF_all", type = int, default =0,\
            help = "add Consistency loss to all combinations")      
        parser.add_argument("--LAMBDA3", type = float,  default=1e-4, \
            help = "the weight of Consistency loss")

        parser.add_argument("--server", type =str, default="wisteria", help="server type (wisteria, colab)")
        parser.add_argument("--network", type =str, default="resnet50", \
            help="the main segmentation model: choose the model from 'resnet50', 'resnet101', or 'vgg16'. ")
        parser.add_argument("--test_crop", type=int, default=0, \
            help = "evaliate model with the target domain without reshape (just random crop)")
        parser.add_argument("--G_use_ref", type=int, default=0, \
            help = "0: Mask generator takes source images, 1: takes source and reference images")

        parser.add_argument("--pb_set", type=str, default="region", \
            help="region or resolution")
        parser.add_argument("--down_scale", type=float, default=0.25, \
            help="down scale ratio")

        parser.add_argument("--prob_imagenet_ref", type=float, default=0.0, \
            help = "the probability to use the imagenet for reference, 0:not use imagenet for ref")
        parser.add_argument("--imgnet_path", type= str, default="/work/gk36/share/ILSVRC2015/Data/CLS-LOC/train/")

        opt = parser.parse_args()
        return opt

    
    def show_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            #message += '{:<15}: {:<30}{}\n'.format(str(k), str(v), comment)
            #message += str(k)+" "*int(30-len(str(k))*1.5) + ": " + str(v)+ " "*int(30-len(str(v))*1.5) + "\n"
            if str(k) == "imgnet_path":
                continue
            elif str(k) in ["optimize", "randomize", "use_only_ref", "MFI", "fullmask",\
                 "halfmask", "COF","COF_all", "test_crop", "final"]:
                message += '{:<15}: {:<30}\n'.format(str(k), str(bool(v)))
            else:
                message += '{:<15}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)
