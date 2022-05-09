import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import *

class BatchNorm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum=0.99, is_training=True, is_conv_out=True, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.is_training = is_training
            self.is_conv_out = is_conv_out
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, inputs):
        shape = inputs.get_shape().as_list()
        batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(shape)-1)), name='moments')
        ema_apply_op = self.ema.apply([batch_mean, batch_var])
        self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

        beta = tf.identity(tf.Variable(name="beta", initial_value=np.zeros([shape[-1]]), dtype=tf.float32))
        gamma = tf.identity(tf.Variable(name="gamma",
            initial_value=(np.sqrt(2./(9*64)))*np.random.normal(size=shape[-1]), dtype=tf.float32))

        # print(beta.name, tf.Session().run(beta))

        if self.is_training:
            with tf.control_dependencies([ema_apply_op]):
                mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, self.epsilon)
        # normed = tf.nn.batch_norm_with_global_normalization(
        #         x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

        # beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        # scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        # pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        # pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        #
        # if self.is_training:
        #     if self.is_conv_out:
        #         batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        #     else:
        #         batch_mean, batch_var = tf.nn.moments(inputs, [0])
        #
        #     train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        #     train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        #     with tf.control_dependencies([train_mean, train_var]):
        #         return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        # else:
        #     return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) + (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


# def conv2d(input_, output_channels, f_h=3, f_w=3, d_h=1, d_w=1, stddev=(np.sqrt(2./(9*64))), name="conv2d", with_w=False):
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [f_h, f_w, input_.get_shape()[-1], output_channels],
#                             initializer=tf.random_normal_initializer(stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
#         biases = tf.get_variable('biases', output_channels, initializer=tf.constant_initializer(0.0))
#         conv   = tf.nn.bias_add(conv, biases)
#         if with_w:
#             return conv, w, biases
#         else:
#             return conv

# def conv2d_relu(input_, output_channels, f_h=3, f_w=3, d_h=1, d_w=1, stddev=(np.sqrt(2./(9*64))), name="conv2d", with_w=False):
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [f_h, f_w, input_.get_shape()[-1], output_channels],
#                             initializer=tf.random_normal_initializer(stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
#         biases = tf.get_variable('biases', output_channels, initializer=tf.constant_initializer(0.0))
#         conv   = tf.nn.relu(tf.nn.bias_add(conv, biases))
#         if with_w:
#             return conv, w, biases
#         else:
#             return conv

def conv3d(input_, output_channels, f_d=3, f_h=5, f_w=5, d_f=1, d_h=2, d_w=2, stddev=0.02, name="conv3d", with_w=False):
    with tf.variable_scope(name):
        # filter : [filter_depth, height, width, in_channels, output_channels]
        w = tf.get_variable('w', [f_d, f_h, f_w, input_.get_shape()[-1], output_channels],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_f, d_h, d_w, 1], padding='SAME')
        #input = tf.get_variable('input', [1, 4, 30, 30, 1],
                            #initializer=tf.random_normal_initializer(stddev=stddev))
        #w = tf.get_variable('w', [4, 3, 3, 1, 64],
                            #initializer=tf.random_normal_initializer(stddev=stddev))

        #xx = tf.nn.conv3d(input, w, strides=[1, 1, 1, 1, 1], padding='VALID')

        biases = tf.get_variable('biases', output_channels, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases
        else:
            return conv


def recon1(input_, name="recon1"):

    # with sub-pixel
    with tf.variable_scope(name):
        h0_1rec, h0_wrec, h0_brec = conv3d(input_, 12,
                                f_d=4, f_h=3, f_w=3, d_f=4, d_h=1, d_w=1, name=name, with_w=True)

        h0_1rec = tf.squeeze(h0_1rec, axis=1)
        h0_rec = phase_shift(h0_1rec, 3, 2, False)
        h0_rec = tf.nn.relu(h0_rec)

        return h0_rec, h0_1rec, h0_wrec, h0_brec


def recon2(input_, name="recon2"):

    # with sub-pixel
    with tf.variable_scope(name):
        h0_1rec, h0_wrec, h0_brec = conv2d(input_, 12,  with_w=True)
        h0_1rec = tf.nn.relu(h0_1rec)
        h0_rec = phase_shift(h0_1rec, 3, 2, False)
        return h0_rec, h0_1rec,  h0_wrec, h0_brec



def deconv2d(input_, output_shape, f_h=5, f_w=5, d_h=1, d_w=1, stddev=0.02, name="deconv2d", with_w=False):
    """define the 2d filters
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [f_h, f_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def deconv3d(input_, output_shape, f_d=3, f_h=5, f_w=5, d_f=1, d_h=1, d_w=1, stddev=0.02, name="deconv3d", with_w=False):
    """define the 3d filters
    """
    with tf.variable_scope(name):
        # filter : [filter_depth, height, width, output_channels, in_channels]
        w = tf.get_variable('w', [f_d, f_h, f_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_f, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(input_, leak=0.2, name="lrelu"):
    """
    if x>0, lrelu(x)=x
    else lrelu(x)=leak*x
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input_ + f2 * abs(input_)




def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
