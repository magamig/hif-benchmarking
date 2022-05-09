# HSISR-Spectral-Auxiliary-Heterogeneous-Datasets-PyTorch
This is the implementation of our paper **Hyperspectral Image Super-Resolution with Spectral Mixup and Heterogeneous Datasets**, which is available on arXiv:  https://arxiv.org/abs/2101.07589. Our network is based on SSPSR of which the code is available on github: https://github.com/junjun-jiang/SSPSR.

**Abstract**: This  work  studies  Hyperspectral  image  (HSI)  super-resolution   (SR).   HSI   SR   is   characterized   by   high-dimensional data and a limited amount of training exam-ples.   This  exacerbates  the  undesirable  behaviors  of  neu-ral networks such as memorization and sensitivity to out-of-distribution samples.   This work addresses these issueswith three contributions. First, we propose a simple, yet ef-fective data augmentation routine, termed Spectral Mixup,to construct effective virtual training samples.  Second, weobserve that HSI SR and RGB image SR are correlated anddevelop a novel multi-tasking network to train them jointlyso that the auxiliary task RGB image SR can provide addi-tional supervision. Finally, we extend the network to a semi-supervised setting so that it can learn from datasets contain-ing low-resolution HSIs only. With these contributions, ourmethod is able to learn from heterogeneous datasets and liftthe requirement for having a large amount of HD HSI train-ing samples.  Extensive experiments on four datasets showthat our method outperforms existing methods significantlyand underpin the relevance of our contributions. 

## Datasets and data processing
Cave, Harvard, NTIRE2020, DIV2K, and Chikusei have been used in our paper. The code for data processing can be found in folder **matlab_code**. Three matlab scripts including **generate_train_data.m**, **generate_test_data.m**, and **crop_image.m** are used for generating image patches. Because the Chikusei dataset consist of only one big image of 2517×2335 pixels, we need to crop it into training images and test images. The folder **Chikusei** contains all matlab scripts for generating Chikusei images. 

## Requirement
Python 3.8, PyTorch 1.7.0, cuda v10.1.243

## Trained models with conditions: RGBSR + SSL + SMixup
1. **CAVE**: 
    models/Cave_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Wed_Mar_31_03:00:46_2021.pth
2. **Harvard**: 
    models/Harvard_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Fri_Apr__2_15:35:55_2021.pth
3. **NTIRE2020**: 
    models/NTIRE2020_DeepShare_Blocks=3_Subs8_Ovls2_Feats=256_epoch_10_Thu_Apr__1_17:20:30_2021.pth
    
## Training example:
Here we give an example how to train the network: 

`python mains.py train --batch_size
8
--UseRGB
1
--theta_rgb
3
--UseLabeledSpectralMixUp
1
--theta_LabeledSpectralMixUp
1
--UseUnlabelConsistency
1
--theta_unlabel
3
--dataset_name
"Harvard"
--train_dir_mslabel
/Harvard/x4/Train_Semi_Labeled_10_x4/
--data_train_num
3396
--data_test_num
40
--test_dir
/Harvard/x4/Testx4_All/
--train_dir_rgb
/DIV2KRGB/Train_RGB2020_8channels_x4/
--train_dir_msunlabel
/Harvard/x4/Train_Semi_Unlabeled_30_x4/
--save_dir
./`

We use the following arguments to control whether to use and how to use our proposed components: 

**--UseRGB 1**:                          (#1 means to use RGB image SR, and 0 otherwise)

**--theta_rgb 3**:                      (#3 means RGB data is 3 times as much as the labeled hyperspectal data)  

**--UseUnlabelConsistency 1**:          (#1 means to use the SSL component based on unlabled consistency, and 0 otherwise) 

**--theta_unlabel 3**:                  (#3 means unlabeled hyperspectral data is 3 times as much as the labeled hyperspectal data)  

**--UseLabeledSpectralMixUp 1**:        (#1 means to use our spectral mixup data augmentation, and 0 otherwise)

**--theta_LabeledSpectralMixUp 1**:     (#1 means to use 1 augmented sample per real sample, 2 or 3...is possible)

## Training images
In order to generate reproducible results and facilitate a direct comparison, we also release our splits of all datasets for labeled training sets, unlabeled training sets, and test sets. The lists for CAVE, Harvard, and NTIRE2020 can be found in folder **image_list**. 



## Citation
```
@InProceedings{HSISRLi21,
  author = {Li, Ke and Dai, Dengxin and Konukoglu, Ender and {Van Gool}, Luc},
  title = {Hyperspectral Image Super-Resolution with Spectral Mixup and Heterogeneous Datasets},
  booktitle = {ArXiv:2101.07589},
  year = 2021
}```
