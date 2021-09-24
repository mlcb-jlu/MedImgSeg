#!/bin/bash
export PATH=/home/weidu/anaconda3/envs/CBFNet/bin:$PATH

chmod -R 777 results
python -u main.py --light True --dataset ISIC --folder isic_1 --resume False --phase test
chmod -R 777 results

chmod -R 777 results
python -u main.py --light True --dataset BraTS --folder brats_1 --resume False --phase test
chmod -R 777 results


