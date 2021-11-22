#!/bin/bash
export PATH=/home/weidu/anaconda3/envs/CBFNet/bin:$PATH #Change to your own virtual environment path

#start Dens-CRF processing on ISIC dataset
python -u infer_cls_test_our20210321.py --dataset ISIC --folder isic_1 --phase val
python -u infer_cls_test_our20210321.py --dataset ISIC --folder isic_1 --phase test

#start Dens-CRF processing on BraTS dataset
python -u infer_cls_test_our20210321.py --dataset BraTS --folder brats_1 --phase val
python -u infer_cls_test_our20210321.py --dataset BraTS --folder brats_1 --phase test



