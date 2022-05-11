#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lry
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class rnn_model():
    def __init__(self,opt):
        self.opt = opt
        self.rnn_cell_local = rnn.GRUCell(self.opt.hidden_size_local, name='rnn_cell_local')
        self.rnn_cell_global_fw = rnn.GRUCell(self.opt.hidden_size_global, name='rnn_cell_global_fw')
        self.rnn_cell_global_bw = rnn.GRUCell(self.opt.hidden_size_global, name='rnn_cell_global_bw')

        self.enc_W_q = self.weight_variable([self.opt.hidden_size_global*2 , self.opt.enc_q])
        self.enc_W_k = self.weight_variable([self.opt.hidden_size_global*2 , self.opt.enc_k])
        self.enc_W_v = self.weight_variable([self.opt.hidden_size_global*2 , self.opt.enc_v])

        self.dec_W_q = self.weight_variable([self.opt.gener_hidden_size*2 , self.opt.dec_q])
        self.enc_W_k_z = self.weight_variable([self.opt.hidden_size_global , self.opt.enc_k_z])
        self.enc_W_v_z = self.weight_variable([self.opt.hidden_size_global , self.opt.enc_v_z])

        self.w2 = self.weight_variable([self.opt.f_1, self.opt.f_1, self.opt.dimXg, self.opt.filter_num_1])
        self.b2 = self.bias_variable([self.opt.filter_num_1])
        self.w3 = self.weight_variable([self.opt.f_2, self.opt.f_2, self.opt.filter_num_1, self.opt.filter_num_2])
        self.b3 = self.bias_variable([self.opt.filter_num_2])
        self.w4 = self.weight_variable([self.opt.f_3, self.opt.f_3, self.opt.filter_num_2, self.opt.filter_num_3])
        self.b4 = self.bias_variable([self.opt.filter_num_3])
        self.w5 = self.weight_variable([self.opt.dimX,self.opt.dimXg*self.opt.hidden_size_local])
        self.b5 = self.bias_variable([self.opt.dimXg*self.opt.hidden_size_local])
        self.rnn_cell_xg_fw = rnn.GRUCell(self.opt.hidden_size_global, name='rnn_cell_xg_fw')
        self.rnn_cell_xg_bw = rnn.GRUCell(self.opt.hidden_size_global, name='rnn_cell_xg_bw')

        self.w6 = self.weight_variable([self.opt.hidden_size_global * self.opt.dimXg*2, self.opt.dimZ])
        self.b6 = self.bias_variable([self.opt.dimZ])
        self.w7 = self.weight_variable([self.opt.hidden_size_global * self.opt.dimXg*2, self.opt.dimZ])
        self.b7 = self.bias_variable([self.opt.dimZ])
        self.w9 = self.weight_variable([self.opt.dimXg*self.opt.hidden_size_global*2, self.opt.dimZ])
        self.b9 = self.bias_variable([self.opt.dimZ])
        self.w10 = self.weight_variable([self.opt.dimXg*self.opt.hidden_size_global*2, self.opt.dimZ])
        self.b10 = self.bias_variable([self.opt.dimZ])
        # self.rnn_cell_gener = rnn.LSTMCell(self.opt.gener_hidden_size, name='LSTM_gener')

        self.rnn_cell_gener_fw = rnn.LSTMCell(self.opt.gener_hidden_size, name='rnn_cell_gener_fw')
        self.rnn_cell_gener_bw = rnn.LSTMCell(self.opt.gener_hidden_size, name='rnn_cell_gener_bw')

        self.w_gener_0 = self.weight_variable([self.opt.dec_q, self.opt.sep[0]])
        self.w_gener_1 = self.weight_variable([self.opt.dec_q, self.opt.sep[1]-self.opt.sep[0]])
        self.w_gener_2 = self.weight_variable([self.opt.dec_q, self.opt.sep[2]-self.opt.sep[1]])
        self.w_gener_3 = self.weight_variable([self.opt.dec_q, self.opt.sep[3]-self.opt.sep[2]])
        self.b_gener_0 = self.weight_variable([self.opt.sep[0]])
        self.b_gener_1 = self.weight_variable([self.opt.sep[1]-self.opt.sep[0]])
        self.b_gener_2 = self.weight_variable([self.opt.sep[2]-self.opt.sep[1]])
        self.b_gener_3 = self.weight_variable([self.opt.sep[3]-self.opt.sep[2]])

        return

    def inference(self,x_low,x_rgb,x_bic):
        sep = self.opt.sep
        hidden_size_local = self.opt.hidden_size_local
        outputs_1_xl, state_1_xl = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_low[:, 0:sep[0]], [self.opt.num, sep[0], 1]), dtype=tf.float32)

        outputs_2_xl, state_2_xl = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_low[:, sep[0]:sep[1]], [self.opt.num, sep[1]-sep[0], 1]), dtype=tf.float32)

        outputs_3_xl, state_3_xl = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_low[:, sep[1]:sep[2]], [self.opt.num, sep[2]-sep[1], 1]), dtype=tf.float32)

        outputs_4_xl, state_4_xl = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_low[:, sep[2]:sep[3]], [self.opt.num, sep[3]-sep[2], 1]), dtype=tf.float32)


        global_input_xl = tf.transpose(tf.reshape(tf.concat(
            [outputs_1_xl[:, -1, :], outputs_2_xl[:, -1, :], outputs_3_xl[:, -1, :], outputs_4_xl[:, -1, :]], 1), [self.opt.num, hidden_size_local, self.opt.dimXg]), [0, 2, 1])


        outputs_global_xl, state_global_xl = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_global_fw, self.rnn_cell_global_bw, inputs=global_input_xl, dtype=tf.float32)
        outputs_global_xl = tf.reshape(tf.concat([outputs_global_xl[0],outputs_global_xl[1]], 2), [self.opt.num, -1])

        #############  compute the self-attention of encoder ##############
        outputs_global_xl_2d = tf.reshape(outputs_global_xl, [self.opt.num * self.opt.dimXg, self.opt.hidden_size_global * 2])
        self.enc_xl_query = tf.reshape(tf.matmul(outputs_global_xl_2d, self.enc_W_q),[self.opt.num, self.opt.dimXg, self.opt.enc_q])
        self.enc_xl_key = tf.reshape(tf.matmul(outputs_global_xl_2d, self.enc_W_k), [self.opt.num , self.opt.dimXg, self.opt.enc_k])
        self.enc_xl_value = tf.reshape(tf.matmul(outputs_global_xl_2d, self.enc_W_v), [self.opt.num , self.opt.dimXg, self.opt.enc_v])
        self.att_xl_en = self.qk_Attention(self.enc_xl_query, self.enc_xl_key)
        outputs_global_xl = tf.reshape(tf.matmul(self.att_xl_en, self.enc_xl_value), [self.opt.num, -1])

        # fusion
        x_bic_reshape = tf.reshape(x_bic, [-1, self.opt.dimX])
        # h1 = tf.nn.tanh(self.conv2d(x_bic_reshape, self.w1) + self.b1)

        outputs_1_xb, state_1_xb = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_bic_reshape[:, 0:sep[0]], [self.opt.batch_size, sep[0], 1]), dtype=tf.float32)

        outputs_2_xb, state_2_xb = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_bic_reshape[:, sep[0]:sep[1]], [self.opt.batch_size, sep[1]-sep[0], 1]), dtype=tf.float32)

        outputs_3_xb, state_3_xb = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_bic_reshape[:, sep[1]:sep[2]], [self.opt.batch_size, sep[2]-sep[1], 1]), dtype=tf.float32)

        outputs_4_xb, state_4_xb = tf.nn.dynamic_rnn(self.rnn_cell_local, inputs=tf.reshape(x_bic_reshape[:, sep[2]:sep[3]], [self.opt.batch_size, sep[3]-sep[2], 1]), dtype=tf.float32)


        global_input_xb = tf.transpose(tf.reshape(tf.concat(
            [outputs_1_xb[:, -1, :], outputs_2_xb[:, -1, :], outputs_3_xb[:, -1, :], outputs_4_xb[:, -1, :]], 1), [self.opt.batch_size, hidden_size_local, self.opt.dimXg]), [0, 2, 1])

        outputs_global_xb, state_global_xb = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_global_fw, self.rnn_cell_global_bw, inputs=global_input_xb, dtype=tf.float32)
        outputs_global_xb = tf.reshape(tf.concat([outputs_global_xb[0],outputs_global_xb[1]], 2), [self.opt.batch_size, -1])

        outputs_1_xm = tf.nn.tanh(self.conv2d(x_rgb, self.w2) + self.b2)
        outputs_2_xm = tf.nn.tanh(self.conv2d(outputs_1_xm, self.w3) + self.b3)
        outputs_3_xm = tf.nn.tanh(self.conv2d(outputs_2_xm, self.w4) + self.b4)

        outputs_global_xm = tf.reshape(outputs_3_xm, [self.opt.batch_size, -1])
        outputs_global_xh = 0.5 * outputs_global_xb + outputs_global_xm  # try adding a parameter

        #############  compute the self-attention of encoder ##############
        outputs_global_xh_2d = tf.reshape(outputs_global_xh, [self.opt.batch_size * self.opt.dimXg, self.opt.hidden_size_global * 2])
        self.enc_xh_query = tf.reshape(tf.matmul(outputs_global_xh_2d, self.enc_W_q),[self.opt.batch_size, self.opt.dimXg, self.opt.enc_q])
        self.enc_xh_key = tf.reshape(tf.matmul(outputs_global_xh_2d, self.enc_W_k), [self.opt.batch_size , self.opt.dimXg, self.opt.enc_k])
        self.enc_xh_value = tf.reshape(tf.matmul(outputs_global_xh_2d, self.enc_W_v), [self.opt.batch_size , self.opt.dimXg, self.opt.enc_v])
        self.att_xh_en = self.qk_Attention(self.enc_xh_query, self.enc_xh_key)
        outputs_global_xh = tf.reshape(tf.matmul(self.att_xh_en, self.enc_xh_value), [self.opt.batch_size, -1])


        return outputs_global_xl, outputs_global_xh

    def generator(self, eps1, eps2, x_low, x_rgb, x_bic, outputs_xl, outputs_xg, Trans):

        Z1_mu = tf.matmul(outputs_xl, self.w6) + self.b6
        Z1_log_sigma = 0.5 * (tf.matmul(outputs_xl, self.w7) + self.b7)
        Z1 = Z1_mu + tf.exp(Z1_log_sigma) * eps1

        Z2_mu = tf.matmul(outputs_xg, self.w9) + self.b9
        Z2_log_sigma = 0.5 * (tf.matmul(tf.reshape(outputs_xg, [self.opt.batch_size, -1]), self.w10) + self.b10)
        Z2 = Z2_mu + tf.exp(Z2_log_sigma) * eps2

        Z1 = tf.reshape(Z1,[self.opt.num, self.opt.dimXg, -1])
        Z2 = tf.reshape(Z2,[self.opt.batch_size, self.opt.dimXg, -1])

        #############  compute the key and value of encoder ##############
        Z1_2d = tf.reshape(Z1, [self.opt.num * self.opt.dimXg, self.opt.hidden_size_global])
        self.enc_xl_key_zl = tf.reshape(tf.matmul(Z1_2d, self.enc_W_k_z), [self.opt.num , self.opt.dimXg, self.opt.enc_k_z])
        self.enc_xl_value_zl = tf.reshape(tf.matmul(Z1_2d, self.enc_W_v_z), [self.opt.num , self.opt.dimXg, self.opt.enc_v_z])

        #############  compute the key and value of xb encoder ##############
        Z2_2d = tf.reshape(Z2, [self.opt.batch_size * self.opt.dimXg, self.opt.hidden_size_global])
        self.enc_xh_key_zh = tf.reshape(tf.matmul(Z2_2d, self.enc_W_k_z), [self.opt.batch_size , self.opt.dimXg, self.opt.enc_k_z])
        self.enc_xh_value_zh = tf.reshape(tf.matmul(Z2_2d, self.enc_W_v_z), [self.opt.batch_size , self.opt.dimXg, self.opt.enc_v_z])

        outputs_gener_xl, state_gener_xl = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_gener_fw, self.rnn_cell_gener_bw, inputs=Z1, dtype=tf.float32)
        outputs_gener_xh, state_gener_xh = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_gener_fw, self.rnn_cell_gener_bw, inputs=Z2, dtype=tf.float32)

        outputs_gener_xl = tf.concat([outputs_gener_xl[0],outputs_gener_xl[1]], 2)
        outputs_gener_xh = tf.concat([outputs_gener_xh[0],outputs_gener_xh[1]], 2)


        # #############  compute the self-attention of encoder ##############
        # outputs_gener_xl_2d = tf.reshape(outputs_gener_xl, [self.opt.num * self.opt.dimXg, self.opt.gener_hidden_size * 2])
        # self.dec_xl_query = tf.reshape(tf.matmul(outputs_gener_xl_2d, self.dec_W_q),[self.opt.num, self.opt.dimXg, self.opt.dec_q])
        # self.dec_xl_key = tf.reshape(tf.matmul(outputs_gener_xl_2d, self.dec_W_k), [self.opt.num , self.opt.dimXg, self.opt.dec_k])
        # self.dec_xl_value = tf.reshape(tf.matmul(outputs_gener_xl_2d, self.dec_W_v), [self.opt.num , self.opt.dimXg, self.opt.dec_v])
        # self.att_xl_de = self.qk_Attention(self.dec_xl_query, self.dec_xl_key)
        # output_add_atten_xl = tf.matmul(self.att_xl_de, self.dec_xl_value)
        #
        # #############  compute the self-attention of encoder ##############
        # outputs_gener_xh_2d = tf.reshape(outputs_gener_xh, [self.opt.batch_size * self.opt.dimXg, self.opt.gener_hidden_size * 2])
        # self.dec_xh_query = tf.reshape(tf.matmul(outputs_gener_xh_2d, self.dec_W_q),[self.opt.batch_size, self.opt.dimXg, self.opt.dec_q])
        # self.dec_xh_key = tf.reshape(tf.matmul(outputs_gener_xh_2d, self.dec_W_k), [self.opt.batch_size , self.opt.dimXg, self.opt.dec_k])
        # self.dec_xh_value = tf.reshape(tf.matmul(outputs_gener_xh_2d, self.dec_W_v), [self.opt.batch_size , self.opt.dimXg, self.opt.dec_v])
        # self.att_xh_de = self.qk_Attention(self.dec_xh_query, self.dec_xh_key)
        # output_add_atten_xh = tf.matmul(self.att_xh_de, self.dec_xh_value)



        gener_xl = tf.reshape(outputs_gener_xl,[self.opt.num * self.opt.dimXg, self.opt.gener_hidden_size*2])
        self.dec_xl_query = tf.reshape(tf.matmul(gener_xl, self.dec_W_q),[self.opt.num, self.opt.dimXg, self.opt.dec_q])
        self.att_xl_de = self.qk_Attention(self.dec_xl_query, self.enc_xl_key_zl)
        output_add_atten_xl = tf.matmul(self.att_xl_de, self.enc_xl_value_zl)


        gener_xh = tf.reshape(outputs_gener_xh,[self.opt.batch_size * self.opt.dimXg, self.opt.gener_hidden_size*2])
        self.dec_xh_query = tf.reshape(tf.matmul(gener_xh, self.dec_W_q),[self.opt.batch_size, self.opt.dimXg, self.opt.dec_q])
        self.att_xh_de = self.qk_Attention(self.dec_xh_query, self.enc_xh_key_zh)
        output_add_atten_xh = tf.matmul(self.att_xh_de, self.enc_xh_value_zh)


        xl_outputs_0 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xl[:,0,:],[self.opt.num,-1]),self.w_gener_0) + self.b_gener_0)
        xl_outputs_1 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xl[:,1,:],[self.opt.num,-1]),self.w_gener_1) + self.b_gener_1)
        xl_outputs_2 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xl[:,2,:],[self.opt.num,-1]),self.w_gener_2) + self.b_gener_2)
        xl_outputs_3 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xl[:,3,:],[self.opt.num,-1]),self.w_gener_3) + self.b_gener_3)

        xh_outputs_0 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xh[:,0,:],[self.opt.batch_size,-1]),self.w_gener_0) + self.b_gener_0)
        xh_outputs_1 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xh[:,1,:],[self.opt.batch_size,-1]),self.w_gener_1) + self.b_gener_1)
        xh_outputs_2 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xh[:,2,:],[self.opt.batch_size,-1]),self.w_gener_2) + self.b_gener_2)
        xh_outputs_3 = tf.nn.sigmoid(tf.matmul(tf.reshape(output_add_atten_xh[:,3,:],[self.opt.batch_size,-1]),self.w_gener_3) + self.b_gener_3)

        xl_outputs = tf.concat([xl_outputs_0,xl_outputs_1,xl_outputs_2,xl_outputs_3],1)
        xh_outputs = tf.concat([xh_outputs_0,xh_outputs_1,xh_outputs_2,xh_outputs_3],1)

        xg_outputs = tf.matmul(xh_outputs, Trans)

        KL1, KL2 = self.compute_KL(Z1_mu, Z1_log_sigma, Z2_mu, Z2_log_sigma)
        logpxlow, logpxrgb = self.compute_likelihood(x_low, xl_outputs, x_rgb, xg_outputs)
        Loss1 = self.opt.patch_num * (0.001 * KL1 + logpxlow)
        Loss2 = 0.001 * KL2 + logpxrgb
        return Loss1, Loss2, xh_outputs


    def compute_KL(self, Zl_mu, Zl_log_sigma, Zh_mu, Zh_log_sigma):
        KL1 = 0.5 * tf.reduce_sum(1 + 2 * Zl_log_sigma - Zl_mu ** 2 - tf.exp(2 * Zl_log_sigma))
        KL2 = 0.5 * tf.reduce_sum(1 + 2 * Zh_log_sigma - Zh_mu ** 2 - tf.exp(2 * Zh_log_sigma))
        # KL2 = 0.5 * tf.reduce_sum(1 + 2*Z2_log_sigma - (Z2_mu-Z2_prior)**2 - tf.exp(2*Z2_log_sigma))
        return KL1, KL2

    def DOT_Attention(self, outputs, inputs):
        att = tf.nn.softmax(tf.matmul(outputs, tf.transpose(inputs, [0, 2, 1])))
        return att

    def qk_Attention(self, outputs, inputs):
        att = tf.nn.softmax(tf.matmul(outputs, tf.transpose(inputs, [0, 2, 1]))/ np.sqrt(self.opt.enc_q))
        return att

    def compute_likelihood(self, x_low,X_low_mu,x_rgb,X_rgb_mu):

        a0 = tf.Variable(tf.truncated_normal([], 0, self.opt.sigmaInit, dtype=tf.float32))

        logpxlow = tf.reduce_sum(- (0.5 * tf.log(2 * np.pi)) - 0.5 * (
            (tf.reshape(x_low, [self.opt.num, self.opt.dimX]) - X_low_mu)) ** 2)

        logpxrgb= tf.reduce_sum(- (0.5 * tf.log(2 * np.pi)) - 0.5 * (
            (tf.reshape(x_rgb, [self.opt.batch_size, self.opt.dimXg]) - X_rgb_mu) / tf.exp(a0)) ** 2) - self.opt.dimXg * a0

        return logpxlow, logpxrgb

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=self.opt.sigmaInit, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def get_xl_groups(self, x_low):
        ##### seperate the input 200 to 6 groups####
        sep = int(self.opt.dimX / self.opt.dimXg)
        x_group = []
        for index1 in range(self.opt.dimXg):
            if index1 == (self.opt.dimXg - 1):
                x_local = x_low[:, index1 * sep:]
            else:
                x_local = x_low[:, index1 * sep:(index1 + 1) * sep]
            x_group.append(x_local)
        return x_group

    def conv_t(self, x, w, s):
        return tf.nn.conv2d_transpose(x, w, output_shape=s, strides=[1, 1, 1, 1], padding='SAME')
