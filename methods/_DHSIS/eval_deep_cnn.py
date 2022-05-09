# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:44:17 2017

@author: gaj
"""

from __future__ import absolute_import, division
from keras.layers import Input, Conv2D, Activation, BatchNormalization

# %env CUDA_VISIBLE_DEVICES=0
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import h5py
import os
import scipy.io as sio
from keras.callbacks import EarlyStopping

#from resnet import read_data, eval_get_cnn
from deepcnn import read_data, eval_get_cnn

if __name__ == "__main__":
    
    
    inputs, outputs = eval_get_cnn()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights('./deepcnn_res_noise.h5', by_name=True)
    
    for i in range(32):
        
        ind = i+1
        
        print 'processing for %d'%ind

        data = sio.loadmat('./初始化/%d.mat'%ind)
        
        data = data['b']
        
        data = np.expand_dims(data,0)
    
        data_get = model.predict(data, batch_size=1, verbose=1)
        
        data_get = np.reshape(data_get, (512, 512, 31))
        
        data_get = np.array(data_get, dtype=np.float64)
        
        sio.savemat('./get/eval_%d.mat'%ind, {'b': data_get}) 
    

    
    
