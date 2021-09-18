#MedImgSeg

# Consistency label transfer network for weakly supervised medical image segmentation
## Introduction
Current auto-segmentation methods of medical images are
greatly hampered by the insufficient and ambiguous clinical
annotation. Actually, the rough classification labels (e.g. disease or normal) rather than the precise segmentation mask,
are more common and available in clinical practice, but how
to fully utilize those weak clinical labels to infer the precise lesion mask and guide the medical image segmentation remains largely unexplored. In this paper, we proposed
a medical image segmentation framework to generate lesion mask by a class attention map (CAM) guided cycleconsistency label transferring network. This model can simultaneously identify pixel level label-discriminated mask
meanwhile maintain the semantic information and anatomical structure of medical image to precisely define segmentation boundary by collating the label and semantic related
region through an image synthesis procedure. In addition, the
produced lesion mask is further prompted by a joint discrimination strategy for the synthetized and generated image belonging to the opposite category. Extensive experiments of
the proposed method on BraTS and ISIC datasets demonstrate consistently superior performance over existing state-of-the-art methods.

Comparison between the other methods of weakly supervised segmentation and ours. (a) Ground Truth (GT),
(b) Ours, (c) PSA (Ahn and Kwak 2018), (d) SEAM (Wang et al. 2020) and (e) IRN (Ahn, Cho, and Kwak 2019).
<div align=center><img width="1500" height="1200" src="https://raw.githubusercontent.com/mlcb-jlu/MedImgSeg/master/img-folder/weak_result_contrast.png"/></div>

Comparision of the segmentation performance JA, DI, AC, SE and SP (%) of our weakly supervised pipeline on
(a) BraTS dataset and (b) ISIC dataset against CAM (Zhou et al. 2016), CAM+CRF (Zhou et al. 2016), PCM (Wang
et al. 2020), PCM+CRF (Wang et al. 2020), PSA (Ahn and Kwak 2018), SEAM (Wang et al. 2020), IRN (Ahn, Cho, and Kwak 2019)
<div align=center><img width="1500" height="1200" src="https://raw.githubusercontent.com/mlcb-jlu/MedImgSeg/master/img-folder/weak_result_contrast.png"/></div>



This code was made public to share our research for the benefit of the scientific community. Do not use it for immoral purposes.
## Prerequisites
Our code is built based on Pytorch, and the packages to be installed are as follows:
```
conda create -n CBFNet python=3.6.6
conda activate CBFNet
pip install -r requirements.txt
```

## Data Preparation
To evaluate the segmentation performance of different methods, we conducted experiments on two different medical datasets, including BraTS datasets and ISIC datasets.

BraTS image dataset: Both BraTS 2018 and BraTS2019 data were utilized in our experiments. The
image data of these datasets contain 3D MRI volumes with
dimensions of 240 × 240 × 155. In order to adapt to the
network architecture, we processed all the 3D images into
2D slices, then cropped to 1:1 and scaled to 256 × 256. Finally, we randomly divided the data set to obtain 563, 90 and 90 training, validation and testing data.

lSIC image dataset: The lSIC image dataset in our experiments is the 2018 ISIC skin lesion segmentation challenge
dataset [6]. The image size ranges from 771 × 750 to 6748× 4499. Firstly, we removed the images which contains information other 
than the original skin. To balance the segmentation performance and computational cost, we resized
all the images to 256 × 256 using bicubic interpolation. Thehealth data is to intercept healthy areas of different sizes
from the 4 corners of the original diseased image and crop them to 256 × 256. Finally, we randomly divided the data
set to obtain 200, 31 and 31 training, validation and testing data.


We have already trained two models on BraTS dataset and ISIC dataset ,you can download them and the data we have processed for training and test.(location:https://pan.baidu.com/s/1rx29DxWq5W6bTh9NcvT0Tw, password:1111)
Anyway you shound get the dataset and the model folders like:
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
 - results
   - BraTS
     - brats_1
       - model1
   - ISIC
     - isic_1
       - model1
```


## Run
To test our trained model on BraTS dataset and ISIC dataset:
```
python main.py --light True --dataset BraTS --phase test
```
Before you start training, you need to run visdom and open the corresponding address in the browser to monitor each loss:
```
python -m visdom.server
```
To train and validation on BraTS dataset and ISIC dataset:
```
python main.py --light True --dataset BraTS --phase train --stage 1
```


## Citation

If our paper helps your research, please cite it in your publications:
```
@article{
      title={Consistency label transfer network for weakly supervised medical image segmentation}, 
      author={Wei Du{weidu@jlu.edu.cn} and Rixin Zhou（zhourx19@mails.jlu.edu.cn） and Yu Sun{ysun18@mails.jlu.edu.cn} and  
      and Huimin Bao{baohm18@mails.jlu.edu.cn} and Shiyi Tang{sytang20@mails.jlu.edu.cn} and Xuan Zhao and Li, Ying{liying@jlu.edu.cn} and Gaoyang Li{lgyzngc@tongji.edu.cn}},
      year={2021}
}
```

If you have any problem, please feel free to contact me at [weidu@jlu.edu.cn](mailto:weidu@jlu.edu.cn) or raise an issue.
