#!/bin/bash
export PATH=/home/weidu/anaconda3/envs/CBFNet/bin:$PATH

python -u main.py --light True --dataset ISIC --folder isic_1 --resume False --phase test

python -u main.py --light True --dataset BraTS --folder brats_1 --resume False --phase test




