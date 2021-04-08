# wsMedSegV2

# Consistency label transfer network for weakly supervised medical image segmentation
## Introduction
Current auto-segmentation methods of medical images
are hampered by the insufficient and ambiguous clinical annotation. However, rough classification labels (e.g. disease
or normal) are often available in practice, and how to guide
precise lesion mask generation and improve the medical image segmentation using those weak labels remains largely
unexplored. Here, we proposed a weakly-supervised medical image segmentation framework which produces lesion
mask by a class attention mask guided cycle-consistency
label generative network. This model can simultaneously identify the label-significant mask region and maintain
the image semantic information to precisely define segmentation boundary with an unpaired image synthesis procedure. Specifically, the produced lesion mask is fine-tuned
using a teacher-student framework. Extensive experiments
of the proposed method on BraTS and ISIC dataset demonstrate consistently superior performance over existing SOTA methods.

This code was made public to share our research for the benefit of the scientific community. Do NOT use it for immoral purposes.


## Prerequisites
- Install PyTorch (version 1.7.0), Torch Vision and dependencies from http://pytorch.org
- Install requirements.txt (```pip install -r requirements.txt```)

## Data Preparation
To evaluate the segmentation performance of different methods, we conducted experiments on two different medical datasets, including BraTS images datasets and ISIC images datasets.

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

The code requires a directory containing the following files:
- `dataset/BraTS/trainA`: folder with all disease BraTS image for train
- `dataset/BraTS/trainB`: folder with all normal BraTS image for train
- `dataset/BraTS/testA`: folder with all disease BraTS image for test
- `dataset/BraTS/testB`: folder with all normal BraTS image for test
- `dataset/ISIC/trainA`: folder with all disease ISIC image for train
- `dataset/ISIC/trainB`: folder with all normal ISIC image for train
- `dataset/ISIC/testA`: folder with all disease ISIC image for test
- `dataset/ISIC/testB`: folder with all normal ISIC image for test

## Run
To train in BraTS dataset:
```
python main.py --light True --dataset BraTS --phase train --stage 1
```
To test:
```
python main.py --light True --dataset BraTS --phase test
```


