#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lry
"""
import tensorflow as tf
import numpy as np
import time
import os
import random
import opt_rnn_attention as opt
from rnn_attention_model import rnn_model

index1 = np.arange(0, opt.N1, opt.scale)
index2 = np.arange(0, opt.N2, opt.scale)
Xl_big = np.zeros((opt.N1, opt.N2, opt.dimX), dtype='float32')

for r1 in range(len(index1)):
    for r2 in range(len(index2)):
        Xl_big[index1[r1]: index1[r1]+opt.scale, index2[r2]: index2[r2]+opt.scale, :] = opt.Xl_3d[r1, r2, :]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=opt.sigmaInit, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def conv_t(x, w, s):
    return tf.nn.conv2d_transpose(x, w, output_shape=s, strides=[1, 1, 1, 1], padding='SAME')

def average_pool_2x2(x):
    return tf.layers.average_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')

# inputs
x_low = tf.placeholder('float32', [opt.num, opt.dimX])
x_bic = tf.placeholder('float32', [opt.patch_num, opt.patch_size, opt.patch_size, opt.dimX])
x_rgb = tf.placeholder('float32', [opt.patch_num, opt.patch_size, opt.patch_size, opt.dimXg])
x_pri = tf.placeholder('float32', [opt.patch_num, opt.patch_size, opt.patch_size, opt.dimX])
Trans = tf.placeholder('float32', [opt.dimX, opt.dimXg])
eps1 = tf.placeholder('float32',  [opt.num, opt.dimZ])
eps2 = tf.placeholder('float32',  [opt.batch_size, opt.dimZ])

RNN_model = rnn_model(opt)
outputs_global, outputs_xg = RNN_model.inference(x_low,x_rgb,x_bic)
Loss1,Loss2,xh_outputs = RNN_model.generator(eps1,eps2,x_low,x_rgb,x_bic,outputs_global, outputs_xg,Trans)

optimizer1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-Loss1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-Loss1-Loss2)

# joint_loss = -Loss1-Loss2
# joint_tvars = tf.trainable_variables()
# joint_grads, _ = tf.clip_by_global_norm(tf.gradients(joint_loss, joint_tvars), opt.max_grad_norm)
# optimizer = tf.train.AdamOptimizer(opt.learning_rate).apply_gradients(zip(joint_grads, joint_tvars))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
saver = tf.train.Saver(max_to_keep=int(opt.Maxiter/opt.step))
graph = tf.get_default_graph()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) # tf.Graph().as_default()
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with graph.as_default(), tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    print('pretraining ... ...')
    for i in range(opt.Pretrain):
        e0 = np.random.normal(0, 1, [opt.num, opt.dimZ])
        _, loss1 = sess.run([optimizer1, Loss1],
                              feed_dict={x_low: opt.Xl_2d, eps1: e0})
        print("PreIter {}, loss1 = {}".format(i+1, loss1 / opt.num))
    if os.path.exists(opt.path + '/params/checkpoint'):
        saver.restore(sess, opt.path + '/params/model_{}.ckpt'.format(opt.lastTrain))
        print('*********load model*********')
    if not os.path.exists(opt.path):
        os.mkdir(opt.path)
    if not os.path.exists(opt.path+'/params'):
        os.mkdir(opt.path+'/params')
    index1 = np.arange(0, opt.N1-opt.patch_size, opt.gap)  # 0 is included, while N1 is not included
    if index1[-1] != opt.N1 - opt.patch_size:
        index1[-1] = opt.N1 - opt.patch_size
    index2 = np.arange(0, opt.N2-opt.patch_size, opt.gap)  # 0 is included, while N1 is not included
    if index2[-1] != opt.N2 - opt.patch_size:
        index2[-1] = opt.N2 - opt.patch_size
    indexList = []
    for in1 in index1:
        for in2 in index2:
            indexList.append([in1,in2])
    x_low_batch = opt.Xl_2d
    x_bic_batch = np.zeros((opt.patch_num, opt.patch_size, opt.patch_size, opt.dimX), dtype='float32')
    x_rgb_batch = np.zeros((opt.patch_num, opt.patch_size, opt.patch_size, opt.dimXg), dtype='float32')
    x_pri_batch = np.zeros((opt.patch_num, opt.patch_size, opt.patch_size, opt.dimX), dtype='float32')

    for j in range(opt.lastTrain+1, opt.Maxiter+1):
        print('learning ... ...', j)
        random.shuffle(indexList)
        loss_1 = 0
        loss_2 = 0
        begin = time.time()
        for m in range(len(indexList)//opt.patch_num):
            for n in range(opt.patch_num):
                a = indexList[m * opt.patch_num + n][0]
                b = indexList[m * opt.patch_num + n][1]
                x_bic_batch[n, :, :, :] = opt.Xl_bicubic[a:a+opt.patch_size,b:b+opt.patch_size,:]
                x_rgb_batch[n, :, :, :] = opt.Xg_3d[a:a+opt.patch_size,b:b+opt.patch_size,:]
                x_pri_batch[n, :, :, :] = Xl_big[a:a+opt.patch_size,b:b+opt.patch_size,:]
            e1 = np.random.normal(0, 1, [opt.num, opt.dimZ])
            e2 = np.random.normal(0, 1, [opt.batch_size, opt.dimZ])
            _, loss1, loss2 = sess.run([optimizer, Loss1, Loss1 ],
                                                  feed_dict={x_low: x_low_batch, x_bic: x_bic_batch,
                                                             x_rgb: x_rgb_batch, x_pri: x_pri_batch,
                                                             Trans: opt.Trans_data, eps1: e1, eps2: e2})
            loss_1 += loss1
            loss_2 += loss2
        end = time.time()
        print("Iteration {}, LOGP = {} ,time = {}".format(j, (loss_1+loss_2)/opt.batch_size, end-begin))
        print("---- loss1 = {}, loss2 = {}".format(loss_1/opt.num, loss_2/opt.batch_size))
        if j % opt.step == 0:
            saver.save(sess, opt.path + '/params/model_{}.ckpt'.format(j))
        begin = end
