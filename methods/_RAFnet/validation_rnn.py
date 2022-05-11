#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lry
"""
import tensorflow as tf
import numpy as np
import time
import os
import opt_rnn_attention as opt
from rnn_attention_model import rnn_model
import scipy.io as sio

opt.patch_size1 = opt.N1
opt.patch_size2 = opt.N2

opt.patch_num = 1
opt.batch_size = opt.patch_size1 * opt.patch_size2 * opt.patch_num

index1 = np.arange(0, opt.N1, opt.scale)
index2 = np.arange(0, opt.N2, opt.scale)
Xl_big = np.zeros((opt.N1, opt.N2, opt.dimX), dtype='float32')

# inputs
x_low = tf.placeholder('float32', [opt.num, opt.dimX])
x_bic = tf.placeholder('float32', [opt.patch_num, opt.patch_size1, opt.patch_size2, opt.dimX])
x_rgb = tf.placeholder('float32', [opt.patch_num, opt.patch_size1, opt.patch_size2, opt.dimXg])
x_pri = tf.placeholder('float32', [opt.patch_num, opt.patch_size1, opt.patch_size2, opt.dimX])
Trans = tf.placeholder('float32', [opt.dimX, opt.dimXg])
eps1 = tf.placeholder('float32', [opt.num, opt.dimZ])
eps2 = tf.placeholder('float32', [opt.batch_size, opt.dimZ])

# fusion
# x_bic_reshape = tf.reshape(x_bic, [batch_size, 1, dimX, 1])
# w6 = weight_variable([1, f2_1, 1, 1])
# b6 = bias_variable([1])
# h3 = tf.nn.tanh(conv2d(x_bic_reshape, w6) + b6)
# h42 = tf.reshape(h41, [batch_size, 1, dimX, 1])
# h43 = 0.5*h3 + h42

RNN_model = rnn_model(opt)
outputs_xl, outputs_xg = RNN_model.inference(x_low, x_rgb, x_bic)
Loss1, Loss2, xh_outputs = RNN_model.generator(eps1, eps2, x_low, x_rgb, x_bic, outputs_xl, outputs_xg, Trans)
# att_xl, att_xh, att_xl_2, att_xh_2 = RNN_model.get_att(eps1, eps2, x_low, x_rgb, x_bic, outputs_xl, outputs_xg, Trans)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
saver = tf.train.Saver(max_to_keep=int(opt.Maxiter / opt.step))
graph = tf.get_default_graph()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) # tf.Graph().as_default()
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
iterations = np.arange(100, opt.Maxiter + opt.step, opt.step)
with graph.as_default(), tf.Session(config=config) as sess:
    for iteration in iterations:
        if os.path.exists(opt.path + '/params/checkpoint'):
            saver.restore(sess, opt.path + '/params/model_{}.ckpt'.format(iteration))
            print('*********Begin testing*********')
        else:
            print('*********The model does not exist!*********')

        if not os.path.exists(opt.path + '/RecImg'):
            os.mkdir(opt.path + '/RecImg')
        if not os.path.exists(opt.path + '/attention'):
            os.mkdir(opt.path + '/attention')

        index1 = np.arange(0, opt.N1, opt.patch_size1)
        index2 = np.arange(0, opt.N2, opt.patch_size2)
        begin = time.time()
        for i in range(len(index1)):
            for j in range(len(index2)):
                print("processing i = {}, j = {}".format(i, j))
                x_bic_batch = opt.Xl_bicubic[index1[i]:index1[i] + opt.patch_size1, index2[j]:index2[j] + opt.patch_size2,
                              :]
                x_rgb_batch = opt.Xg_3d[index1[i]:index1[i] + opt.patch_size1, index2[j]:index2[j] + opt.patch_size2, :]
                e1 = np.zeros([opt.num, opt.dimZ])
                e2 = np.zeros([opt.batch_size, opt.dimZ])
                # x_rec = sess.run(
                x_rec, att_xl_2_, att_xh_2_=sess.run(
                    [xh_outputs, RNN_model.att_xh_en, RNN_model.att_xh_de],
                    # [xh_outputs],
                    feed_dict={x_low: opt.Xl_2d,
                               x_bic: np.reshape(x_bic_batch,
                                                 [opt.patch_num, opt.patch_size1, opt.patch_size2, opt.dimX]),
                               x_rgb: np.reshape(x_rgb_batch,
                                                 [opt.patch_num, opt.patch_size1, opt.patch_size2, opt.dimXg]),
                               Trans: opt.Trans_data, eps1: e1, eps2: e2})
                sio.savemat(opt.path + '/attention/attention{}.mat'.format(iteration), {
                                                                  'att_xl_2': np.reshape(att_xl_2_,
                                                                                       [opt.num, opt.dimXg, -1]),
                                                                  'att_xh_2': np.reshape(att_xh_2_,
                                                                                       [opt.batch_size, opt.dimXg, -1])})
                sio.savemat(opt.path + '/RecImg/rec_{}.mat'.format(iteration), {'Out': x_rec})
                end = time.time()
