import tensorflow as tf
import numpy as np
import math
# import matplotlib.pyplot as plt
import scipy.io as sio
import random
import scipy.misc
import os
from tensorflow.python.training import saver
import tensorflow.contrib.layers as ly
from os.path import join as pjoin
from numpy import *
import numpy.matlib
import scipy.ndimage
import csv


class betapan(object):
    def __init__(self, input, lr_rate, p_rate, nLRlevel, nHRlevel, epoch, is_adam,
                 vol_r, sp_r_lsi, sp_r_msi, initp, config):
        # initialize the input and weights matrices
        self.input = input
        self.initlrate = lr_rate
        self.initprate = p_rate
        self.epoch = epoch
        self.nLRlevel = nLRlevel
        self.nHRlevel = nHRlevel
        self.num = input.num
        self.is_adam = is_adam
        self.vol_r = vol_r
        self.sp_r_lsi = sp_r_lsi
        self.sp_r_msi = sp_r_msi
        self.mean_lrhsi = input.mean_lr_hsi
        self.mean_hrmsi = input.mean_hr_msi
        self.dimlr = input.dimLR
        self.dimhr = input.dimHR
        self.input_lr_hsi = input.rcol_lr_hsi
        self.input_hr_msi = input.rcol_hr_msi
        self.input_lr_msi = input.rcol_lr_msi
        self.input_hr_msi_h = np.zeros([input.dimLR[0]*input.dimLR[1],input.num])
        self.sess = tf.Session(config=config)
        self.initp = initp


        with tf.name_scope('inputs'):
            self.lr_hsi = tf.placeholder(tf.float32, [None, self.dimlr[2]], name='lrhsi_input')
            self.hr_msi = tf.placeholder(tf.float32, [None, self.dimhr[2]], name='hrmsi_input')
            self.hr_msi_h = tf.placeholder(tf.float32,[None, input.num], name = 'hrmsi_h')

        with tf.variable_scope('lr_decoder') as scope:
            self.wdecoder = {
                'lr_decoder_w1': tf.Variable(tf.truncated_normal([self.num, self.num],stddev=0.1)),
                'lr_decoder_w2': tf.Variable(tf.truncated_normal([self.num, self.dimlr[2]], stddev=0.1)),
            }

    def compute_latent_vars_break(self, i, remaining_stick, v_samples):
        # compute stick segment
        stick_segment = v_samples[:, i] * remaining_stick
        remaining_stick *= (1 - v_samples[:, i])
        return (stick_segment, remaining_stick)

    def construct_vsamples(self,uniform,wb,hsize):
        concat_wb = wb
        for iter in range(hsize - 1):
            concat_wb = tf.concat([concat_wb, wb], 1)
        v_samples = uniform ** (1.0 / concat_wb)
        return v_samples

    def encoder_uniform_hsi(self,x,reuse=False):
        layer_1 = tf.matmul(x, self.input.srf.T)
        with tf.variable_scope('lr_hsi_uniform') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_11 = tf.contrib.layers.fully_connected(layer_1, self.nLRlevel[0], activation_fn=None)
            stack_layer_11 = tf.concat([x,layer_11], 1)
            layer_12 = tf.contrib.layers.fully_connected(stack_layer_11, self.nLRlevel[1], activation_fn=None)
            stack_layer_12 = tf.concat([stack_layer_11, layer_12], 1)
            layer_13 = tf.contrib.layers.fully_connected(stack_layer_12, self.nLRlevel[2], activation_fn=None)
            stack_layer_13 = tf.concat([stack_layer_12, layer_13], 1)
            layer_14 = tf.contrib.layers.fully_connected(stack_layer_13, self.nLRlevel[3], activation_fn=None)
            stack_layer_14 = tf.concat([stack_layer_13, layer_14], 1)

            uniform = tf.contrib.layers.fully_connected(stack_layer_14, self.num, activation_fn=None)
        return layer_1, uniform

    def encoder_uniform_msi(self,x,reuse=False):
        with tf.variable_scope('hr_msi_uniform') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_11 = tf.contrib.layers.fully_connected(x, self.nLRlevel[0], activation_fn=None)
            stack_layer_11 = tf.concat([x,layer_11], 1)
            layer_12 = tf.contrib.layers.fully_connected(stack_layer_11, self.nLRlevel[1], activation_fn=None)
            stack_layer_12 = tf.concat([stack_layer_11, layer_12], 1)
            layer_13 = tf.contrib.layers.fully_connected(stack_layer_12, self.nLRlevel[2], activation_fn=None)
            stack_layer_13 = tf.concat([stack_layer_12, layer_13], 1)
            layer_14 = tf.contrib.layers.fully_connected(stack_layer_13, self.nLRlevel[3], activation_fn=None)
            stack_layer_14 = tf.concat([stack_layer_13, layer_14], 1)
            # layer_15 = tf.contrib.layers.fully_connected(stack_layer_14, self.nLRlevel[3], activation_fn=None)
            # stack_layer_15 = tf.concat([stack_layer_14, layer_15], 1)
            uniform = tf.contrib.layers.fully_connected(stack_layer_14, self.num, activation_fn=None)
        return layer_11, uniform

    def encoder_beta_hsi(self,x,reuse=False):
        with tf.variable_scope('lr_hsi_beta') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_21 = tf.contrib.layers.fully_connected(x, self.nLRlevel[0], activation_fn=None)
            stack_layer_21 = tf.concat([x,layer_21], 1)
            layer_22 = tf.contrib.layers.fully_connected(stack_layer_21, self.nLRlevel[1], activation_fn=None)
            stack_layer_22 = tf.concat([layer_22, stack_layer_21], 1)
            layer_32 = tf.contrib.layers.fully_connected(stack_layer_22, 1, activation_fn=None)
        return layer_32

    def encoder_beta_msi(self,x,reuse=False):
        with tf.variable_scope('hr_msi_beta') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_21 = tf.contrib.layers.fully_connected(x, self.nLRlevel[0], activation_fn=None)
            stack_layer_21 = tf.concat([x,layer_21], 1)
            layer_22 = tf.contrib.layers.fully_connected(stack_layer_21, self.nLRlevel[1], activation_fn=None)
            stack_layer_22 = tf.concat([layer_22, stack_layer_21], 1)
            layer_32 = tf.contrib.layers.fully_connected(stack_layer_22, 1, activation_fn=None)
        return layer_32

    def encoder_vsamples_hsi(self, x, hsize, reuse=False):
        layer1, uniform = self.encoder_uniform_hsi(x,reuse)
        uniform = tf.nn.sigmoid(uniform)
        wb = self.encoder_beta_hsi(layer1,reuse)
        wb = tf.nn.softplus(wb)
        v_samples = self.construct_vsamples(uniform,wb,hsize)
        return v_samples, uniform, wb

    def encoder_vsamples_msi(self, x, hsize, reuse=False):
        stack_layer_12, uniform = self.encoder_uniform_msi(x,reuse)
        uniform = tf.nn.sigmoid(uniform)
        wb = self.encoder_beta_msi(x,reuse)
        wb = tf.nn.softplus(wb)
        v_samples = self.construct_vsamples(uniform,wb,hsize)
        return v_samples, uniform, wb

    def construct_stick_break(self,vsample, dim, stick_size):
        size = dim[0]*dim[1]
        size = int(size)
        remaining_stick = tf.ones(size, )
        for i in range(stick_size):
            [stick_segment, remaining_stick] = self.compute_latent_vars_break(i, remaining_stick, vsample)
            if i == 0:
                stick_segment_sum_lr = tf.expand_dims(stick_segment, 1)
            else:
                stick_segment_sum_lr = tf.concat([stick_segment_sum_lr, tf.expand_dims(stick_segment, 1)],1)
        return stick_segment_sum_lr


    def encoder_lr_hsi(self, x, reuse=False):
        v_samples,uniform, wb = self.encoder_vsamples_hsi(x, self.num, reuse)
        stick_segment_sum_lr = self.construct_stick_break(v_samples, self.dimlr, self.num)
        return stick_segment_sum_lr


    def encoder_hr_msi(self, x, reuse=False):
        v_samples,v_uniform, v_beta = self.encoder_vsamples_msi(x, self.num, reuse)
        stick_segment_sum_msi = self.construct_stick_break(v_samples, self.dimhr, self.num)
        return stick_segment_sum_msi

    def encoder_hr_msi_init(self, x, reuse=False):
        v_samples,v_uniform, v_beta = self.encoder_vsamples_msi(x, self.num, reuse)
        stick_segment_sum_msi_init = self.construct_stick_break(v_samples, self.dimlr, self.num)
        return stick_segment_sum_msi_init

    def decoder_hsi(self, x):
        layer_1 = tf.matmul(x, self.wdecoder['lr_decoder_w1'])
        layer_2 = tf.matmul(layer_1, self.wdecoder['lr_decoder_w2'])
        return layer_2

    def decoder_msi(self,x):
        layer_1 = tf.matmul(x, self.wdecoder['lr_decoder_w1'])
        layer_2 = tf.matmul(layer_1, self.wdecoder['lr_decoder_w2'])
        layer_3 = tf.add(layer_2,self.input.mean_lr_hsi)
        return layer_3

    def gen_lrhsi(self, x, reuse=False):
        encoder_op = self.encoder_lr_hsi(x, reuse)
        decoder_op = self.decoder_hsi(encoder_op)
        return decoder_op

    def gen_hrmsi(self, x, reuse=False):
        encoder_op = self.encoder_hr_msi(x, reuse)
        decoder_hr = self.decoder_msi(encoder_op)
        decoder_op = tf.matmul(decoder_hr,self.input.srf.T)
        decoder_plus_m = tf.add(decoder_op, -self.input.mean_hr_msi)
        # decoder_sphere = tf.matmul(decoder_plus_m,self.input.invsig_msi)
        return decoder_plus_m

    def gen_hrhsi(self, x, reuse=True):
        encoder_op = self.encoder_hr_msi(x, reuse)
        decoder_hr = self.decoder_msi(encoder_op)
        return decoder_hr

    def next_feed(self):
        feed_dict = {self.hr_msi:self.input_hr_msi, self.lr_hsi:self.input_lr_hsi}
        return feed_dict

    def gen_msi_h(self, x, reuse = False):
        encoder_init = self.encoder_hr_msi_init(x,reuse)
        return encoder_init

    def build_model(self):
        # build model for lr hsi
        y_pred_lrhsi = self.gen_lrhsi(self.lr_hsi, False)
        y_true_lrhsi = self.lr_hsi
        error_lrhsi = y_pred_lrhsi - y_true_lrhsi
        lrhsi_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_lrhsi, 2),0))
        #lrhsi_loss_euc = tf.reduce_mean(tf.pow(error_lrhsi, 2))

        # geometric constraints
        decoder_op = tf.matmul(self.wdecoder['lr_decoder_w1'], self.wdecoder['lr_decoder_w2'])
        decoder = tf.add(decoder_op, self.input.mean_lr_hsi)
        lrhsi_volume_loss = tf.reduce_mean(tf.matmul(tf.transpose(decoder),decoder))

        # spatial sparse
        eps = 0.00000001
        lrhsi_top = self.encoder_lr_hsi(self.lr_hsi, reuse=True)
        lrhsi_base_norm = tf.reduce_sum(lrhsi_top, 1, keepdims=True)
        lrhsi_sparse = tf.div(lrhsi_top, (lrhsi_base_norm + eps))
        lrhsi_loss_sparse = tf.reduce_mean(-tf.multiply(lrhsi_sparse, tf.log(lrhsi_sparse + eps)))

        # lr hsi total loss
        lrhsi_loss = lrhsi_loss_euc + self.vol_r * lrhsi_volume_loss + self.sp_r_lsi * lrhsi_loss_sparse

        # for lr
        theta_basic_decoder = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_decoder')
        theta_uniform_lrhsi = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_hsi_uniform')
        theta_beta_lrhsi = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_hsi_beta')
        counter_lrhsi = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_lrhsi = ly.optimize_loss(loss=lrhsi_loss, learning_rate=self.initlrate,
                                 optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_basic_decoder+theta_uniform_lrhsi+theta_beta_lrhsi,global_step=counter_lrhsi)

        # build model for high resolution msi image
        y_pred_hrmsi = self.gen_hrmsi(self.hr_msi, False)
        y_true_hrmsi = self.hr_msi
        error_hrmsi = y_pred_hrmsi - y_true_hrmsi
        hrmsi_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_hrmsi, 2), 0))
        #hrmsi_loss_euc = tf.reduce_mean(tf.pow(error_hrmsi, 2))

        # spatial sparse
        hrmsi_top = self.encoder_hr_msi(self.hr_msi, reuse=True)
        hrmsi_base_norm = tf.reduce_sum(hrmsi_top, 1, keepdims=True)
        hrmsi_sparse = tf.div(hrmsi_top, (hrmsi_base_norm + eps))
        hrmsi_loss_sparse = tf.reduce_mean(-tf.multiply(hrmsi_sparse, tf.log(hrmsi_sparse + eps)))


        theta_uniform_hrmsi = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='hr_msi_uniform')
        theta_beta_hrmsi = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='hr_msi_beta')

        # abundance init
        msi_h = self.gen_msi_h(self.hr_msi,True)
        error_init = msi_h  - self.hr_msi_h
        msih_init_loss = tf.reduce_mean(tf.pow(error_init, 2))
        counter_init = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_init = ly.optimize_loss(loss=msih_init_loss, learning_rate=self.initlrate,
                                       optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                       variables=theta_uniform_hrmsi+theta_beta_hrmsi,
                                       global_step=counter_init)

        # spectral loss
        nom_pred = tf.reduce_sum(tf.pow(msi_h, 2),0)
        nom_true = tf.reduce_sum(tf.pow(self.hr_msi_h, 2),0)
        nom_base = tf.sqrt(tf.multiply(nom_pred, nom_true))
        nom_top  = tf.reduce_sum(tf.multiply(msi_h,self.hr_msi_h),0)
        angle = tf.reduce_mean(tf.acos(tf.div(nom_top, (nom_base + eps))))
        angle_loss = tf.div(angle,3.1416)  # spectral loss
        counter_angle = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_angle = ly.optimize_loss(loss=angle_loss, learning_rate=self.initlrate,
                                       optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                       variables=theta_uniform_hrmsi+theta_beta_hrmsi,
                                       global_step=counter_angle)


        hrmsi_loss = hrmsi_loss_euc + self.sp_r_lsi * hrmsi_loss_sparse
        # hrmsi_loss = hrmsi_loss_euc


        counter_hrmsi = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_hrmsi = ly.optimize_loss(loss=hrmsi_loss, learning_rate=self.initlrate,
                               optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                               variables= theta_uniform_hrmsi+theta_beta_hrmsi,
                               global_step=counter_hrmsi)



        return lrhsi_loss, opt_lrhsi, hrmsi_loss, opt_hrmsi, lrhsi_volume_loss, lrhsi_loss_sparse, hrmsi_loss_sparse, msih_init_loss, opt_init, angle_loss, opt_angle




    def train(self, load_Path, save_dir, loadLRonly, tol,init_num):

        lrhsi_loss, opt_lrhsi, hrmsi_loss, opt_hrmsi, lrhsi_volume_loss, lrhsi_loss_sparse, hrmsi_loss_sparse, msih_init_loss, opt_h_init,angle_loss, opt_angle = self.build_model()

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if os.path.exists(load_Path):
            if loadLRonly:
                # load part of the variables
                vars = tf.contrib.slim.get_variables_to_restore()
                variables_to_restore = [v for v in vars if v.name.startswith('lr_decoder/')] \
                                        + [v for v in vars if v.name.startswith('lr_hsi_uniform/')] \
                                        + [v for v in vars if v.name.startswith('lr_hsi_beta/')] \
                                        + [v for v in vars if v.name.startswith('hr_msi_uniform/')] \
                                        + [v for v in vars if v.name.startswith('hr_msi_beta/')]
                saver = tf.train.Saver(variables_to_restore)
                load_file = tf.train.latest_checkpoint(load_Path)
                if load_file==None:
                    print('No checkpoint was saved.')
                else:
                    saver.restore(self.sess,load_file)
            else:
                # load all the variables
                saver = tf.train.Saver()
                load_file = tf.train.latest_checkpoint(load_Path)
                if load_file==None:
                    print('No checkpoint was saved.')
                else:
                    saver.restore(self.sess, load_file)
        else:
            saver = tf.train.Saver()

        results_file_name = pjoin(save_dir,"sb_" + "lrate_" + str(self.initlrate)+ ".txt")
        # results_ckpt_name = pjoin(save_dir,"sb_" + "lrate_" + str(self.initlrate)+ ".ckpt")
        results_file = open(results_file_name, 'a')
        feed_dict = self.next_feed()


        sam_hr = 10
        sam_lr = 10
        rate_decay = 0.99977
        count = 0
        stop_cont = 0
        sam_total = zeros(self.epoch+1)
        rmse_total = zeros(self.epoch+1)
        sam_total[0] = 50
        rmse_total[0] = 50

        for epoch in range(self.epoch):
            if sam_lr > tol:
                _, lr_loss = self.sess.run([opt_lrhsi,lrhsi_loss], feed_dict=feed_dict)
                self.initlrate = self.initlrate * rate_decay
                self.vol_r = self.vol_r * rate_decay
                self.sp_r_lsi = self.vol_r * rate_decay

            if (epoch + 1) % 50 == 0:
                # Report and save progress.

                results = "epoch {}: LR HSI loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, lr_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n")
                results_file.flush()

                v_loss = self.sess.run(lrhsi_volume_loss)
                results = "epoch {}: volumn loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, v_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n")
                results_file.flush()

                sp_hsi_loss = self.sess.run(lrhsi_loss_sparse, feed_dict=feed_dict)
                results = "epoch {}: lr sparse loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, sp_hsi_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

                img_lr = self.sess.run(self.gen_lrhsi(self.lr_hsi, reuse=True), feed_dict=feed_dict) + self.mean_lrhsi
                rmse_lr, sam_lr = self.evaluation(img_lr,self.input.col_lr_hsi,'LR HSi',epoch,results_file)


            if (epoch + 1) % 1000 == 0:
                # saver = tf.train.Saver()
                results_ckpt_name = pjoin(save_dir,
                                          "epoch_" + str(epoch) + "_sam_" + str(round(sam_hr, 3)) + ".ckpt")
                save_path = saver.save(self.sess, results_ckpt_name)

                results = "weights saved at epoch {}"
                results = results.format(epoch)
                print(results)
                print('\n')

            if sam_lr <= tol:
                if count == 0:
                    self.input_hr_msi_h = self.sess.run(self.encoder_lr_hsi(self.lr_hsi, True), feed_dict=feed_dict)

                if self.initp == True:
                    while self.initp and count < init_num:
                        _, initloss = self.sess.run([opt_h_init,msih_init_loss],
                                                    feed_dict={self.hr_msi:self.input_lr_msi,
                                                               self.hr_msi_h:self.input_hr_msi_h})
                        initpanloss = "epoch {}: initloss of the msi: {:.9f}"
                        initpanloss = initpanloss.format(count,initloss)
                        print (initpanloss)
                        results_file.write(initpanloss + "\n")
                        results_file.flush()
                        count = count + 1
                        if (count) % 1000 == 0:
                            saver = tf.train.Saver()
                        if initloss<0.00001:
                            self.initp = False

                _, msi_loss = self.sess.run([opt_hrmsi,hrmsi_loss], feed_dict=feed_dict)

                self.initprate = self.initprate * rate_decay
                self.sp_r_msi = self.sp_r_msi * rate_decay

                if (epoch + 1) % 20 == 0:
                    # Report and save progress.
                    results = "epoch {}: HR MSI loss {:.12f} learing_rate {:.9f}"
                    results = results.format(epoch, msi_loss, self.initprate)
                    print(results)
                    print("\n")
                    results_file.write(results + "\n\n")
                    results_file.flush()

                    sp_msi_loss = self.sess.run(hrmsi_loss_sparse, feed_dict=feed_dict)
                    results = "epoch {}: hr sparse loss {:.12f} learing_rate {:.9f}"
                    results = results.format(epoch, sp_msi_loss, self.initprate)
                    print(results)
                    print("\n")
                    results_file.write(results + "\n\n")
                    results_file.flush()

                    _, angleloss = self.sess.run([opt_angle, angle_loss], feed_dict={self.hr_msi: self.input_lr_msi,
                                                                                     self.hr_msi_h: self.input_hr_msi_h})
                    angle = "Angle of the pan: {:.12f}"
                    angle = angle.format(angleloss)
                    print(angle)
                    results_file.write(angle + "\n")
                    results_file.flush()

                    # img_hr = self.sess.run(self.gen_hrmsi(self.hr_msi, reuse=True), feed_dict=feed_dict) + self.mean_hrmsi
                    # sam_hr = self.evaluation(img_hr,self.input.col_hr_msi,'HR MSI',epoch,results_file)
                    img_hr = self.sess.run(self.gen_hrhsi(self.hr_msi, reuse=True), feed_dict=feed_dict)
                    rmse_hr, sam_hr = self.evaluation(img_hr,self.input.col_hr_hsi,'HR MSI',epoch,results_file)
                    stop_cont = stop_cont + 1
                    sam_total[stop_cont] = sam_hr
                    rmse_total[stop_cont] = rmse_hr
                    if ((sam_total[stop_cont-1] / sam_total[stop_cont]) < 1 - 0.0001 and (rmse_total[stop_cont-1]/rmse_total[stop_cont]<1 - 0.0001)):
                        results_ckpt_name = pjoin(save_dir,"epoch_" + str(epoch) + "_sam_" + str(round(sam_hr, 3)) + ".ckpt")
                        save_path = saver.save(self.sess, results_ckpt_name)
                        print('training is done')
                        break;

        return save_path

    def evaluation(self,img_hr,img_tar,name,epoch,results_file):
        # evalute the results
        ref = img_tar*255.0
        tar = img_hr*255.0
        lr_flags = tar<0
        tar[lr_flags]=0
        hr_flags = tar>255.0
        tar[hr_flags] = 255.0

        #ref = ref.astype(np.int)
        #tar = tar.astype(np.int)

        diff = ref - tar;
        size = ref.shape
        rmse = np.sqrt( np.sum(np.sum(np.power(diff,2))) / (size[0]*size[1]));
        # rmse_list.append(rmse)
        # print('epoch '+str(epoch)+' '+'The RMSE of the ' + name + ' is: '+ str(rmse))
        results = name + " epoch {}: RMSE  {:.12f} "
        results = results.format(epoch,  rmse)
        print (results)
        results_file.write(results + "\n")
        results_file.flush()

        # spectral loss
        nom_top = np.sum(np.multiply(ref, tar),0)
        nom_pred = np.sqrt(np.sum(np.power(ref, 2),0))
        nom_true = np.sqrt(np.sum(np.power(tar, 2),0))
        nom_base = np.multiply(nom_pred, nom_true)
        angle = np.arccos(np.divide(nom_top, (nom_base)))
        angle = np.nan_to_num(angle)
        sam = np.mean(angle)*180.0/3.14159
        # sam_list.append(sam)
        # print('epoch '+str(epoch)+' '+'The SAM of the ' + name + ' is: '+ str(sam)+'\n')
        results = name + " epoch {}: SAM  {:.12f} "
        results = results.format(epoch,  sam)
        print (results)
        print ("\n")
        results_file.write(results + "\n")
        results_file.flush()
        return rmse, sam

    def generate_hrhsi(self, save_dir, filename):
        # self.sess.run(tf.global_variables_initializer())

        gen_hrhsi = self.gen_hrhsi(self.hr_msi, reuse=False)
        feed_dict = self.next_feed()

        saver = tf.train.Saver()
        save_path = tf.train.latest_checkpoint(filename)
        # save_path = filename
        if save_path == None:
            print('No checkpoint was saved.')
        else:
            saver.restore(self.sess, save_path)
            print(save_path + '  is loaded.')


        # save hidden layers
        hrhsi = self.sess.run(gen_hrhsi, feed_dict=feed_dict)
        np.savetxt(save_dir + "/hrhsi.csv", hrhsi, delimiter=",")






