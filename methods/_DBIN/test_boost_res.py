# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:56:26 2019

@author: ww
"""


import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import cv2
import os
from mat_convert_to_tfrecord_p_end import read_and_decode_test
from skimage.measure import compare_ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def fusion_net(Z, Y, num_spectral = 31, num_fm = 128, num_ite=5, reuse=False):
    
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
            Xi = ly.conv2d(Xt, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="res"+str(i)+"1")
            Xi = ly.conv2d(Xi, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="res"+str(i)+"2")
            Xt = Xt + Xi    
        X = ly.conv2d(Xt, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                       weights_regularizer = ly.l2_regularizer(weight_decay), 
                       weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.leaky_relu, reuse = tf.AUTO_REUSE, scope="out")
    
        return X     

        
def compute_ms_ssim(image1, image2):
    image1 = np.reshape(image1, (512,512,31))
    image2 = np.reshape(image2, (512,512,31))
    n = image1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = compare_ssim(image1[:,:,i], image2[:,:,i])
        ms_ssim += single_ssim
    return ms_ssim/n

def compute_sam(image1, image2):
    image1 = np.reshape(image1, (512*512, 31))
    image2 = np.reshape(image2, (512*512, 31))
    mole = np.sum(np.multiply(image1, image2), axis=1)
    image1_norm = np.sqrt(np.sum(np.square(image1), axis=1))
    image2_norm = np.sqrt(np.sum(np.square(image2), axis=1))
    deno = np.multiply(image1_norm, image2_norm)

    sam = np.rad2deg(np.arccos((mole+10e-12)/(deno+10e-12)))
    return np.mean(sam)
    
def compute_ergas(mse, out):
    out = np.reshape(out, (512*512,31))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (31, 1))
    out_mean = np.reshape(out_mean, (31, 1))
    ergas = 100/8*np.sqrt(np.mean(mse/out_mean**2))
    
    return ergas
    
def get_edge(data):  
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape)==3:
            rs[i,:,:] = data[i,:,:] - cv2.boxFilter(data[i,:,:],-1,(5,5))
        else:
            rs[i,:,:,:] = data[i,:,:,:] - cv2.boxFilter(data[i,:,:,:],-1,(5,5))
    return rs

if __name__=='__main__':

    test_data = './training_data/testp.tfrecords'
    
    batch_size = 1
    weight_decay = 2e-5
    model_directory = './models_boost_res/'
    image_size = 512
    
    tf.reset_default_graph()
      
    # placeholder for tensor
    gt_holder = tf.placeholder(dtype = tf.float32,shape = [batch_size,image_size,image_size,31])
    ms_holder = tf.placeholder(dtype = tf.float32,shape = [batch_size,image_size//8,image_size//8,31])
    pan_holder = tf.placeholder(dtype = tf.float32,shape = [batch_size,image_size,image_size,3])
    pan2_holder = tf.placeholder(dtype = tf.float32,shape = [batch_size,image_size//2,image_size//2,3])
    pan4_holder = tf.placeholder(dtype = tf.float32,shape = [batch_size,image_size//4,image_size//4,3])
    
    pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch = read_and_decode_test(test_data, batch_size=batch_size)

    X = fusion_net(pan_holder, ms_holder)
    
    output = tf.clip_by_value(X,0,1) # final output

    mse = tf.square(output - gt_holder)
    mse = tf.reshape(mse, [512,512,31])
    final_mse = tf.reduce_mean(mse, reduction_indices=[0,1])    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    average_psnr = 0.0  
    average_ssim = 0.0
    average_sam = 0.0
    average_ergas = 0.0    
    
    gt_out = np.zeros(shape = [12, 512, 512,31], dtype = np.float32)
    net_out = np.zeros(shape = [12, 512, 512,31], dtype = np.float32)
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
        

        for i in range(12):
            
            pan, pan2, pan4, gt, ms, ms2 = sess.run([pan_batch, pan2_batch, pan4_batch, gt_batch, ms_batch, ms2_batch])
            out, mse_loss = sess.run([output, final_mse],feed_dict = {pan_holder: pan, pan2_holder:pan2,
                                     pan4_holder:pan4, gt_holder: gt, ms_holder:ms})
            
            gt_out[i,:,:,:]= gt
            net_out[i,:,:,:] = np.array(out)
            
            mse_loss = np.array(mse_loss)
            gt = np.array(gt)
            ms_psnr = np.mean(10*np.log10(1/mse_loss))
            temp_ssim = compute_ms_ssim(out, gt)
            temp_sam = compute_sam(out, gt)
            temp_ergas = compute_ergas(mse_loss, out)
#            temp_ergas = np.sqrt(np.mean(mse_loss/np.reshape(np.mean(np.mean(out, axis = 1), axis=1), (1,31))**2))*100/8
            print("image%d temp_psnr: %.4f,  temp_ssim: %.4f, temp_sam: %.4f, temp_ergas: %.4f" % (i, ms_psnr, temp_ssim, temp_sam, temp_ergas))
            average_psnr += ms_psnr/12.0
            average_ssim += temp_ssim/12.0
            average_sam += temp_sam/12.0
            average_ergas += temp_ergas/12.0
        coord.request_stop()
        coord.join(threads)
        sess.close()
        sio.savemat('./result/out.mat', {'gt_out':gt_out, 'net_out':net_out})
        print("average_psnr: %.4f, average_ssim: %.4f, average_sam: %.4f, average_ergas: %.4f" %(average_psnr, average_ssim, average_sam, average_ergas))
            
