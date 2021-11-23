

#MedImgSeg



# Consistency Label-activated Region Transferring Network for Weakly Supervised Medical Image Segmentation
## Introduction
Official pytorch implementation of paper "Consistency Label-activated Region Transferring Network for Weakly
Supervised Medical Image Segmentation".
This code was made public to share our research for the benefit of the scientific community. Do not use it for immoral purposes.
## Requirements
* python 3.6
* pytorch 1.7.0, torchvision 0.8.1
* CUDA 11.1
* 1 X GPU (24GB)
## Prerequisites
Our code is built based on Pytorch, and the packages to be installed are as follows:
```
sudo git clone https://github.com/mlcb-jlu/MedImgSeg.git
cd MedImgSeg/
```

```
conda create -n CBFNet python=3.6.6
source activate
conda activate CBFNet
```
Install the pytorch you need on the pytorch official website:https://pytorch.org/get-started/locally/. 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
Then install the dependencies: (Anaconda is recommended.)

```
pip install pillow tqdm tensorboardx pyyaml visdom opencv-python nibabel libsvm matplotlib
```


## Data Preparation
To evaluate the segmentation performance of different methods, we conducted experiments on two different medical datasets, including BraTS datasets and ISIC datasets.
You can download for training and testing.(dataset location:https://pan.baidu.com/s/1rx29DxWq5W6bTh9NcvT0Tw, password:1111)

* Anyway you should get the dataset folders like:
```
your_project_location
 - dataset
   - BraTS
     - brats_1
       - 563_A_weak
       - 563_B_weak
       - testA
         - images
         - labels
       - testB
       - val
         - images
         - labels
       - few_sample
         - stage2_50
           - A
           - B
   - ISIC
     - ...
```
## Quickly Test
You can download the trained models we provide and put them in the trained model directory for quickly test.
Run demo.sh to test the model segmentation performance, and the segmentation results are saved in the result directory. (model location:https://pan.baidu.com/s/1oWbK0j5Xl6E2MUQU6TCRzg, password:1111)

```
bash ./demo.sh
```

## Training and Testing
* Step1：you need to run visdom and open the corresponding address in the browser to monitor each loss:
```
python -m visdom.server
```
* Step2： you should change the PATH in the executable bash files to your virtual environment location.
To training , validating and testing on BraTS dataset and ISIC dataset:
```
bash ./start_train.sh
bash ./start_test.sh
bash ./start_crf.sh
```
* Then you will get the result folders like:
```
your_project_location
 - results
   - BraTS
     - brats_1
       - model1 # Storage location for trained models
       - val_folder
         - crf_deal1
           - 1~8 # The segmentation results after dense-CRF post-processing on validation set
           - results.txt # The best segmentation results of the model on the validation set
       - crf_deal1
         - 1~8 # The segmentation results after dense-CRF post-processing on test set
         - results.txt  # The best segmentation results of the model on the test set
   - ISIC
     - ..
```

