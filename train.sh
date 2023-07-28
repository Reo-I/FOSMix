#!/bin/bash
#PJM -g gk36
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM --fs /work,/data
#PJM --mail-list iizuka-reo@g.ecc.u-tokyo.ac.jp
#PJM -j
#PJM -N opt_randomize

module load cuda/11.1
module load pytorch/1.8.1
module load cudnn/8.1.0
module load nccl/2.8.4
source /work/gu15/k36090/my-pytorch-env/bin/activate
#source ~/my-pytorch-env/bin/activate

# move to your working directory on /work/gk36
cd /work/gu14/k36090/opt_randomize

# execute your main program.
python train_unet.py \
    --dataset OEM \
    --n_epochs 150 \
    --ver 50 \
    --seed 0 \
    --learning_rate 0.0001 \
    --randomize 1 \
    --optimize 1 \
    --aug_color 0.5 \
    --LAMBDA1 10.0 \
    --LAMBDA2 0.1 \
    --LAMBDA3 0.00001 \
    --MFI 1 \
    --fullmask 1 \
    --COF 1 \
    --network resnet50 \
    --test_crop 1 \
    --pb_set region \
    --down_scale 0.25 

