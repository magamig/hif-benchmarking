# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:35:34 2019

@author: ww
"""



import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import os
from mat_convert_to_tfrecord_h_test import read_and_decode_test

import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,0'


config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
    

def fusion_net(Z, Y, num_spectral = 31, num_fm = 128, num_ite=3, reuse=False):
    
    with tf.variable_scope('fusion_net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        X = Fusion(Z, Y)
        Xs = X
        
        for i in range(num_ite):
            X = boost_lap(X, Z, Y)
            
            Xs = tf.concat([Xs, X], axis=3)
    
        X = ly.conv2d(Xs, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
    return X

def boost_lap(X, Z_in, Y_in, num_spectral = 31, num_fm = 128, reuse= True):
    
    with tf.variable_scope('recursive'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        Z = ly.conv2d(X, num_outputs = 3, kernel_size = 3, stride = 1, 
                      weights_regularizer = ly.l2_regularizer(weight_decay), 
                      weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu,
                      reuse = tf.AUTO_REUSE, scope='dz')
        
        Y =  ly.conv2d(X, num_outputs = num_spectral, kernel_size = 12, stride = 8, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, 
                          reuse = tf.AUTO_REUSE, scope='dy')
    
        dZ = Z_in - Z
        dY = Y_in - Y
    
        dX = Fusion(dZ, dY)
        X = X + dX
    
    
    return X
    
    
def Fusion(Z, Y, num_spectral = 31, num_fm = 128, reuse = True):
    
    with tf.variable_scope('py'):        
        if reuse:
            tf.get_variable_scope().reuse_variables() 
            
        lms = ly.conv2d_transpose(Y,num_spectral,12,8,activation_fn = None,
                                   weights_initializer = ly.variance_scaling_initializer(), 
                                   weights_regularizer = ly.l2_regularizer(weight_decay), reuse = tf.AUTO_REUSE, scope="lms")
        Xin = tf.concat([lms,Z], axis=3)
        Xt = ly.conv2d(Xin, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="in")
        for i in range(4):
            Xi = ly.conv2d(Xt, num_outputs = num_fm, kernel_size = 3, stride = 1, rate=1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay),
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="res"+str(i)+"1")
            Xi = ly.conv2d(Xi, num_outputs = num_fm, kernel_size = 3, stride = 1, rate=2,
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="res"+str(i)+"2")
            Xt = Xt + Xi    
        X = ly.conv2d(Xt, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="out")
    
        return X
        


if __name__=='__main__':

    start = datetime.datetime.now()
    
    test_data = './training_data/testh.tfrecords'
    
    weight_decay = 2e-5
#    model_directory = './models_lap_end6/'

    model_directory = './models_boost_res_h/'
    
    tf.reset_default_graph()
      
    # placeholder for tensor
    gt_holder = tf.placeholder(dtype = tf.float32,shape = [1,1040,1392,31])
    ms_holder = tf.placeholder(dtype = tf.float32,shape = [1,130,174,31])
    pan_holder = tf.placeholder(dtype = tf.float32,shape = [1,1040,1392,3])
    
    pan_batch, gt_batch, ms_batch = read_and_decode_test(test_data, batch_size=1)

    X = fusion_net(pan_holder, ms_holder)
    
    output = tf.clip_by_value(X,0,1) # final output

    mse = tf.square(output - gt_holder) 
    

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
    with tf.Session() as sess:  
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # loading  model       
        if tf.train.get_checkpoint_state(model_directory):  
           ckpt = tf.train.latest_checkpoint(model_directory)
           saver.restore(sess, ckpt)
           print ("load new model")

        else:
           ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
           saver.restore(sess,ckpt.model_checkpoint_path) # this model uses 128 feature maps and for debug only                                                                   
           print ("load pre-trained model")                            
        

        for i in range(20):
            
            pan, gt, ms = sess.run([pan_batch, gt_batch, ms_batch])
            out = sess.run([output],feed_dict = {pan_holder: pan, gt_holder: gt, ms_holder:ms})
            print(i)
            gt = np.array(gt)
            gt_out = gt
            net_out = np.array(out)
            sio.savemat('./result/gt_%s.mat'%(i), {'output':gt_out})
            sio.savemat('./result/net_%s.mat'%(i), {'output':net_out})
            
            
        coord.request_stop()
        coord.join(threads)
        sess.close()
        
        end = datetime.datetime.now()
        print (end-start)
            
