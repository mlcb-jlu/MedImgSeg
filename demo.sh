#!/bin/bash
export PATH=/home/weidu/anaconda3/envs/CBFNet/bin:$PATH 
#Change to your own virtual environment path

#quickly test on ISIC dataset
python -u main.py --light True --dataset ISIC --folder isic_1 --resume False --phase test
python -u infer_cls_test_our.py --dataset ISIC --folder isic_1 --phase test

#quickly test on BraTS dataset
python -u main.py --light True --dataset BraTS --folder brats_1 --resume False --phase test
python -u infer_cls_test_our.py --dataset BraTS --folder brats_1 --phase test


