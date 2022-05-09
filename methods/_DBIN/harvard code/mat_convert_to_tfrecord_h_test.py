# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:02:09 2018

@author: ww
"""


import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,0'

def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(save_dir, name):
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    
    for j in range(1,31):
        train_dir = './training_data/train_h/train' + str(j) + '.mat'
        train_mat = sio.loadmat(train_dir)
        gt = train_mat['gt'][...]    ## ground truth N*H*W*C
        gt = np.array(gt,dtype = np.float32)
        gt2 = train_mat['gt2'][...]    ## ground truth N*H*W*C
        gt2 = np.array(gt2,dtype = np.float32)
        gt4 = train_mat['gt4'][...]    ## ground truth N*H*W*C
        gt4 = np.array(gt4,dtype = np.float32)
        ms = train_mat['ms'][...]    ## ground truth N*H*W*C
        ms = np.array(ms,dtype = np.float32)
        pan = train_mat['pan'][...]  #### Pan image N*H*W
        pan = np.array(pan, dtype = np.float32)
        pan2 = train_mat['pan2'][...]  #### Pan image N*H*W
        pan2 = np.array(pan2, dtype = np.float32)
        pan4 = train_mat['pan4'][...]  #### Pan image N*H*W
        pan4 = np.array(pan4, dtype = np.float32)
        
        for i in range(0, 256):
             try:
                 pan_i = pan[i,:,:,:] # type(image) must be array!
                 pan2_i = pan2[i,:,:,:]
                 pan4_i = pan4[i,:,:,:]
                 gt_i = gt[i,:,:,:]
                 gt2_i = gt2[i,:,:,:]
                 gt4_i = gt4[i,:,:,:]
                 ms_i = ms[i,:,:,:]
                 pan_raw = pan_i.tostring()
                 pan2_raw = pan2_i.tostring()
                 pan4_raw = pan4_i.tostring()
                 gt_raw = gt_i.tostring()
                 gt2_raw = gt2_i.tostring()
                 gt4_raw = gt4_i.tostring()
                 ms_raw = ms_i.tostring()
                 example = tf.train.Example(features=tf.train.Features(feature={
                         'pan_raw': bytes_feature(pan_raw),
                         'pan2_raw': bytes_feature(pan2_raw),
                         'pan4_raw': bytes_feature(pan4_raw),
                         'gt_raw': bytes_feature(gt_raw),
                         'gt2_raw': bytes_feature(gt2_raw),
                         'gt4_raw': bytes_feature(gt4_raw),
                         'ms_raw': bytes_feature(ms_raw),
                         }))
                 writer.write(example.SerializeToString())
             except IOError as e:
                 print('Could not read:', pan[i])
                 print('error: %s' %e)
                 print('Skip it!\n')
        print(j)
    writer.close()
    print('Transform done!')
    
def convert_to_tfrecord_test(save_dir, name):
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    
    for i in range(1,21):
        print(i)
        train_dir = './training_data/test_h/test' + str(i) + '.mat'
        train_mat = sio.loadmat(train_dir)
        gt = train_mat['gt'][...]    ## ground truth N*H*W*C
        gt = np.array(gt,dtype = np.float32)
        ms = train_mat['ms'][...]    ## ground truth N*H*W*C
        ms = np.array(ms,dtype = np.float32)
        pan = train_mat['pan'][...]  #### Pan image N*H*W
        pan = np.array(pan, dtype = np.float32)
        
        try:
            pan_i = pan # type(image) must be array!
            gt_i = gt
            ms_i = ms
            pan_raw = pan_i.tostring()

            gt_raw = gt_i.tostring()
            ms_raw = ms_i.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                         'pan_raw': bytes_feature(pan_raw),
                         'gt_raw': bytes_feature(gt_raw),
                         'ms_raw': bytes_feature(ms_raw)
                         }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', pan[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
#%%
def read_and_decode_test(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'pan_raw': tf.FixedLenFeature([], tf.string),
                                               'gt_raw': tf.FixedLenFeature([], tf.string),
                                               'ms_raw': tf.FixedLenFeature([], tf.string)
                                               })
    pan = tf.decode_raw(img_features['pan_raw'], tf.float32)
    pan = tf.reshape(pan, [1040, 1392, 3])
    gt = tf.decode_raw(img_features['gt_raw'], tf.float32)
    gt = tf.reshape(gt, [1040,1392,31])
    ms = tf.decode_raw(img_features['ms_raw'], tf.float32)
    ms = tf.reshape(ms, [130,174,31])
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.  
    pan_batch, gt_batch, ms_batch = tf.train.batch([pan, gt, ms],
                                                batch_size = batch_size,
                                                num_threads = 4,
                                                capacity = 300,
                                                allow_smaller_final_batch=False)
    return pan_batch, gt_batch, ms_batch

def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'pan_raw': tf.FixedLenFeature([], tf.string),
                                               'pan2_raw': tf.FixedLenFeature([], tf.string),
                                               'pan4_raw': tf.FixedLenFeature([], tf.string),
                                               'gt_raw': tf.FixedLenFeature([], tf.string),
                                               'gt2_raw': tf.FixedLenFeature([], tf.string),
                                               'gt4_raw': tf.FixedLenFeature([], tf.string),
                                               'ms_raw': tf.FixedLenFeature([], tf.string),
                                               })
    pan = tf.decode_raw(img_features['pan_raw'], tf.float32)
    pan = tf.reshape(pan, [64, 64, 3])
    pan2 = tf.decode_raw(img_features['pan2_raw'], tf.float32)
    pan2 = tf.reshape(pan2, [32, 32, 3])
    pan4 = tf.decode_raw(img_features['pan4_raw'], tf.float32)
    pan4 = tf.reshape(pan4, [16, 16, 3])
    gt = tf.decode_raw(img_features['gt_raw'], tf.float32)
    gt = tf.reshape(gt, [64,64,31])
    gt2 = tf.decode_raw(img_features['gt2_raw'], tf.float32)
    gt2 = tf.reshape(gt2, [32,32,31])
    gt4 = tf.decode_raw(img_features['gt4_raw'], tf.float32)
    gt4= tf.reshape(gt4, [16,16,31])
    ms = tf.decode_raw(img_features['ms_raw'], tf.float32)
    ms = tf.reshape(ms, [8,8,31])
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.  
    pan_batch, pan2_batch, pan4_batch, gt_batch, gt2_batch, gt4_batch, ms_batch = tf.train.shuffle_batch([pan, pan2, pan4, gt, gt2, gt4, ms],
                                                batch_size= batch_size,
                                                capacity = 300,
                                                min_after_dequeue = 200,
                                                allow_smaller_final_batch=False)
    return pan_batch, pan2_batch, pan4_batch, gt_batch,gt2_batch, gt4_batch, ms_batch

'''    
#%% Convert data to TFRecord
save_dir = './training_data/'
BATCH_SIZE = 32

name = 'testh'

#convert_to_tfrecord(save_dir, name)
#convert_to_tfrecord_test(save_dir, name)

def plot_images(pan):
    
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(pan[i,:,:,:])
    plt.show()


tfrecords_file = './training_data/testh.tfrecords'
#pan_batch, pan2_batch, pan4_batch, gt_batch, gt2_batch, gt4_batch, ms_batch = read_and_decode(tfrecords_file, batch_size=4)
pan_batch, gt_batch, ms_batch = read_and_decode_test(tfrecords_file, batch_size=4)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config)  as sess:
    
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            # just plot one batch size            
            pan, gt = sess.run([pan_batch, gt_batch])
            plot_images(pan)
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''