#!/bin/bash
export PATH=/home/weidu/anaconda3/envs/CBFNet/bin:$PATH 
#Change to your own virtual environment path

#start training
chmod -R 777 results
python -u main.py --light True --dataset ISIC --folder isic_1 --resume False --iteration 7000 --lr 0.0001 --adv_weight 1 --cycle_weight 5 --identity_weight 50 --cam_weight 1400 --recon_weight 10 --phase train
chmod -R 777 results
#start validation
python -u main.py --light True --dataset ISIC --folder isic_1 --resume False --phase val
chmod -R 755 results

chmod -R 755 results
python -u main.py --light True --dataset BraTS --folder brats_1 --resume False --iteration 7000 --lr 0.0001 --adv_weight 1 --cycle_weight 10 --identity_weight 10 --cam_weight 1000 --recon_weight 10 --phase train
chmod -R 755 results
python -u main.py --light True --dataset BraTS --folder brats_1 --resume False --phase val
chmod -R 755 results



