# wsMedSegV2

# Consistency label transfer network for weakly supervised medical image segmentation
## Introduction
Official implementation of 《Consistency label transfer network for weakly supervised medical image segmentation》. 

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

- `imgs/`: folder with all image
- `aus_openface.pkl`: dictionary containing the images action units.
- `train_ids.csv`: file containing the images names to be used to train.
- `test_ids.csv`: file containing the images names to be used to test.

An example of this directory is shown in `sample_dataset/`.

To generate the `aus_openface.pkl` extract each image Action Units with [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) and store each output in a csv file the same name as the image. Then run:
```
python data/prepare_au_annotations.py
```

## Run
To train:
```
bash launch/run_train.sh
```
To test:
```
python test --input_path path/to/img
```

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{Pumarola_ijcv2019,
    title={GANimation: One-Shot Anatomically Consistent Facial Animation},
    author={A. Pumarola and A. Agudo and A.M. Martinez and A. Sanfeliu and F. Moreno-Noguer},
    booktitle={International Journal of Computer Vision (IJCV)},
    year={2019}
}
```
