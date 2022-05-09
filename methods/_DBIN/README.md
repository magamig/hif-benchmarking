# Deep-Blind-Hyperspectral-Image-Fusion

This repository is for DBIN and EDBIN introduced in the following papers：

[1] Wu Wang, Weihong Zeng, Yue Huang, Xinghao Ding and John Paisley, "Deep Blind Hyperspectral Image Fusion", ICCV 2019

[2] Wu Wang, Weihong Zeng, Liyan Sun, Ronghui Zhan, Yue Huang, and Xinghao Ding, "Enhanced Deep Blind Hyperspectral Image Fusion", TNNLS 2021 (The code of EDBIN will be available soon）

The code is built on Tensorflow and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA8.0, cuDNN5.1) with 1080Ti GPUs. If you have any issues, please feel free to contact me. My mail : 23320170155546@stu.xmu.edu.cn

## Introduction

Hyperspectral image fusion (HIF) reconstructs high spatial
resolution hyperspectral images from low spatial resolution
hyperspectral images and high spatial resolution
multispectral images. Previous works usually assume that
the linear mapping between the point spread functions of
the hyperspectral camera and the spectral response functions
of the conventional camera is known. This is unrealistic
in many scenarios. We propose a method for blind
HIF problem based on deep learning, where the estimation
of the observation model and fusion process are optimized
iteratively and alternatingly during the super-resolution reconstruction.
In addition, the proposed framework enforces
simultaneous spatial and spectral accuracy. Using three
public datasets, the experimental results demonstrate that
the proposed algorithm outperforms existing blind and nonblind
methods.

## Train
For the CAVE dataset, we first convert the image to *.mat* format, and then generate the *tfrecord* file, which can improve the data reading speed. For the CAVE, Harvard, and NTR2018 data sets, we split the image into 64×64 image blocks without any data augmentation.
Unlike the normalization of natural images, we normalize each spectrum of each image to 0 to 1, because some spectral values are very close. You can download the *tfrecord* file of CAVE dataset from [BaiduPan](https://pan.baidu.com/s/17MbNq2sffgI_jbdBuuj6XA), the file extraction code is "psm1". To train the EDBIN，please run "train_cave_edbin.py".
## Test
At the time of testing, we also first converted the image into a tfrecord file. When calculating PSNR, SSIM, SAM, and ERGAS, we used the same code as DHSIS（Deep Hyperspectral Image Sharpening）, here we thank the code provided by Renwei Dian.
## Results
 ![Image text](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion/blob/master/image_folder/CAVE.png)
 
 ![Image text](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion/blob/master/edbin_cave.png)
 
 ![Image text](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion/blob/master/edbin_ps.png)
## Thanks
Our implementation of CARAFE is based on the pytorch version of [XiaLiPKU](https://github.com/XiaLiPKU/CARAFE)， thanks for their wonderful work. The spectral normlization is based on the implementation of [taki0112](https://github.com/taki0112/Spectral_Normalization-Tensorflow). We have verified that SN is beneficial to both supervised and unsupervised HIF.
## Citation
Wang, W.; Zeng, W.; Huang, Y.; Ding, X.; and Paisley, J.
2019. Deep Blind Hyperspectral Image Fusion. In Proceedings
of the IEEE International Conference on Computer Vision,
4150–4159.

Wang, W.; Fu, X.; Zeng, W.; Sun, L.; Zhan, R.; Huang, Y.;
and Ding, X. 2021. Enhanced Deep Blind Hyperspectral
Image Fusion. IEEE Transactions on Neural Networks and
Learning Systems, 1–11
## Acknowledgements

