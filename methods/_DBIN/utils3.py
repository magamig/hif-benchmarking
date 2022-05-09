# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:54:24 2020

@author: ww
"""


import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.layers as ly

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually Iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma * 0.7
        w_norm = tf.reshape(w_norm, w_shape)
    
    return w_norm

def conv(x, channels, weight_decay, kernel=3, stride=1, use_bias=True, reuse=False, scope='conv'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        regularizer = ly.l2_regularizer(weight_decay)
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], 
                                             initializer=ly.variance_scaling_initializer(), regularizer=regularizer)
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='SAME')            
        if use_bias:
                x = tf.nn.bias_add(x, bias)
        return x

def depthwise_conv(x, channels, weight_decay, kernel=3, stride=1, use_bias=True, reuse=False, scope='conv'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        regularizer = ly.l2_regularizer(weight_decay)
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], 1], 
                                             initializer=ly.variance_scaling_initializer(), regularizer=regularizer)
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.depthwise_conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='SAME')            
        if use_bias:
                x = tf.nn.bias_add(x, bias)
        return x
        
def deconv(x, channels, weight_decay, kernel=3, stride=2, use_bias=True, reuse=False, scope='deconv'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x_shape = x.get_shape().as_list()
        regularizer = ly.l2_regularizer(weight_decay)
        output_shape = [x_shape[0], x_shape[1]*stride, x_shape[2]*stride, channels]
        
        w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], 
                                              initializer=ly.variance_scaling_initializer(), regularizer=regularizer)
        x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride,stride,1], padding='SAME')
        
        if use_bias:
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        return x

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.sigmoid(x)
