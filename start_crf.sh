#!/bin/bash
export PATH=/home/dw/anaconda3/envs/sytorch1.7/bin:$PATH #Change to your own virtual environment path

chmod -R 777 results

python -u infer_cls_test_our20210321.py --dataset ISIC --folder isic_1 --phase val
chmod -R 777 results
python -u infer_cls_test_our20210321.py --dataset ISIC --folder isic_1 --phase test
chmod -R 777 results

python -u infer_cls_test_our20210321.py --dataset BraTS --folder brats_1 --phase val
chmod -R 777 results
python -u infer_cls_test_our20210321.py --dataset BraTS --folder brats_1 --phase test
chmod -R 777 results



