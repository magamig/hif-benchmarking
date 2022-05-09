# coding: utf-8
# Script for converting .mat to .tif
#
# Reference: 
# Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation
# Xiuheng Wang, Jie Chen, Qi Wei, Cédric Richard
#
# 2019/05
# Implemented by
# Xiuheng Wang
# xiuheng.wang@mail.nwpu.edu.cn

import numpy as np
import tifffile as tiff
import os
import scipy.io as scio

for i in range(20):
	img = scio.loadmat('./train/' + str(i+1) + '.mat').get('img')
	tiff.imsave('./train/' + str(i+1) + '.tif', img.astype(np.float32))

for i in range(12):
	img = scio.loadmat('./test/' + str(i+1) + '.mat').get('img')
	tiff.imsave('./test/' + str(i+1) + '.tif', img.astype(np.float32))