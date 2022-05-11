#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lry
"""
import scipy.io as sio
import numpy as np

sep = [20,40,61,103]   # please pay attention to this seperation , setted according to the spectral response function of different sensors.
# setting global parameters
patch_size = 16
patch_size1 = 16
patch_size2 = 16

patch_num = 2
batch_size = patch_size*patch_size*patch_num
gap = 12             # the interval between two adjacent patches, must smaller than 'patch_size'
sigmaInit = 0.01

lastTrain = 0       # go on the former training
Pretrain = 500        # the number of iteration for pretraining
Maxiter = 3000       # the max iterations for training # try 2000 3000 4000
step = 100            # save the model every "step" iterations
learning_rate = 0.0001
max_grad_norm = 0.1

# saving path
path = './result_fusion'

filename = "../processed_data/.."

print("Loading data")
data = sio.loadmat(filename)
Xl_3d = data['xl']
Xl_bicubic = data['xl_bicubic'] # this is the bibubic-interpolated xl image acquired with matlab
Xg_3d = data['xg']
scale = data['scale'][0][0]
Trans_data = data['P']
N1,N2,dimX = data['xh'].shape
s1,s2 = data['xl'].shape[0], data['xl'].shape[1]
dimXg = Xg_3d.shape[2]
Xl_2d = np.reshape(Xl_3d,[-1,dimX])
num = s1*s2            # the number of low-resolution pixels

f2_1 = 9             # 1D filter size
f3_1 = 5             # 2D filter size
hidden_size_local = 30
hidden_size_global = 20
gener_hidden_size = 20

enc_q = enc_k = hidden_size_global * 2
enc_v = hidden_size_global * 2

enc_k_z = enc_v_z = 30
dec_q = 30

filter_num_1 = 20
filter_num_2 = 20
filter_num_3 = hidden_size_global*2*dimXg     #### hidden_size_global*2*dimXg
f_1 = 5
f_2 = 3
f_3 = 1

H3_1 = dimX
dimZ = dimXg*hidden_size_global            # dimension of z
