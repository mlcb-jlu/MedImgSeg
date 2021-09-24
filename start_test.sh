#!/bin/bash
export PATH=/home/dw/anaconda3/envs/sytorch1.7/bin:$PATH

chmod -R 755 results
python -u main.py --light True --dataset ISIC --folder isic_1 --resume False --phase test
chmod -R 755 results

chmod -R 755 results
python -u main.py --light True --dataset BraTS --folder brats_1 --resume False --phase test
chmod -R 755 results


