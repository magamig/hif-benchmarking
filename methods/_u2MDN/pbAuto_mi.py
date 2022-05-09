import tensorflow as tf
import numpy as np
import math
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




# Written by Ying Qu, <yqu3@vols.utk.edu>
# This is a demo code for 'Unsupervised and Unregistered Hyperspectral Image Super-Resolution with Mutual Dirichlet-Net'. 
# The code is for research purposes only. All rights reserved.



# add sparsity to the decoder instead of the volume loss
# change the way of adding mean
class betapan(object):
    def __init__(self, input, lr_rate, p_rate, nLRlevel, nHRlevel, epoch, is_adam,
                 lr_vol_r, lr_mi_hsi, lr_s_hsi, config, save_dir):
        # initialize the input and weights matrices
        self.input = input
        self.dim_hsi = input.dimLR_hsi_lr
        self.dim_msi = input.dimHR_msi
        self.dim_hrhsi = [self.dim_msi[0],self.dim_msi[1],self.dim_hsi[-1]]
        self.initlrate = lr_rate
        self.initprate = p_rate
        self.epoch = epoch
        self.nLRlevel = nLRlevel
        self.nHRlevel = nHRlevel
        self.num = input.num
        self.is_adam = is_adam
        self.lr_vol_r = lr_vol_r
        self.lr_mi_hsi = lr_mi_hsi
        self.lr_s_hsi = lr_s_hsi
        self.hr_reuse = False
        self.min_sam = 3
        self.input_hsi = self.input.patch_hsi_lr
        self.input_msi = self.input.patch_msi

        self.rmse_msi = []
        self.sam_msi = []
        self.rmse_lr_hsi = []
        self.sam_lr_hsi = []
        self.rmse_hr_hsi = []
        self.sam_hr_hsi = []

        self.input_hr = np.reshape(self.input.patch_hr, [self.dim_msi[0] * self.dim_msi[1], self.dim_hsi[2]])

        with tf.name_scope('inputs'):
            self.y_hsi = tf.placeholder(tf.float32, [None, None, self.dim_hsi[2]], name='hsi_input')
            self.y_msi = tf.placeholder(tf.float32, [None, None, self.dim_msi[2]], name='msi_input')

        with tf.variable_scope('hsi_project') as scope:
            self.hsi_project = {
                # 'hsi_project_w1': tf.Variable(tf.truncated_normal([self.dim_hsi[-1], self.dim_msi[-1]],stddev=0.1)),
                'hsi_project_w1': tf.Variable(self.input.srf,trainable=False),
            }

        with tf.variable_scope('hsi_decoder') as scope:
            self.whsidecoder = {
                'hsi_decoder_w1': tf.Variable(tf.truncated_normal([self.num, self.num],stddev=0.1)),
                # 'hsi_decoder_w2': tf.Variable(tf.truncated_normal([self.num, self.num*2], stddev=0.1)),
                'hsi_decoder_w2': tf.Variable(tf.truncated_normal([self.num, self.dim_hsi[-1]], stddev=0.1)),
            }

        self.hsi_size = self.num
        self.msi_size = self.num
        self.srfactor = input.srfactor
        self.sess = tf.Session(config=config)

    def compute_latent_vars_break(self, i, remaining_stick, v_samples):
        # compute stick segment
        tmp = v_samples[..., i]
        stick_segment =tmp * remaining_stick
        remaining_stick *= (1 - tmp)
        return (stick_segment, remaining_stick)

    def construct_vsamples(self,uniform,wb,hsize):
        last_dim = len(wb.shape)
        concat_wb = wb
        for iter in range(hsize - 1):
            concat_wb = tf.concat([concat_wb, wb], last_dim-1)
        v_samples = uniform ** (1.0 / concat_wb)
        return v_samples

    def construct_stick_break(self,vsample, dim, stick_size):
        stick_shape = dim[:-1]
        # stick_size = stick_size-1
        remaining_stick = tf.ones(stick_shape, )
        for i in range(stick_size):
            [stick_segment, remaining_stick] = self.compute_latent_vars_break(i, remaining_stick, vsample)
            if i == 0:
                stick_segment_sum_lr = tf.expand_dims(stick_segment, -1)
            else:
                stick_segment_sum_lr = tf.concat([stick_segment_sum_lr, tf.expand_dims(stick_segment, -1)],-1)
            # if i == stick_size - 1:
            #     stick_segment_sum_lr = tf.concat([stick_segment_sum_lr, tf.expand_dims(remaining_stick, -1)],-1)
        return stick_segment_sum_lr

    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # difference from tf 1.3 version to 0.9 version. the tf.layers.dense  --> tf.contrib.layers.fully_connected
    # tf.concat([],1) --> tf.concat(1,[])
    def encoder_vsamples_hsi(self, x, reuse=False):
        with tf.variable_scope('hsi_encoder') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_11 = tf.contrib.layers.fully_connected(x, self.nLRlevel[0], activation_fn=None)
            stack_layer_11 = tf.concat([layer_11,x], -1)
            layer_12 = tf.contrib.layers.fully_connected(stack_layer_11, self.nLRlevel[1], activation_fn=None)
            stack_layer_12 = tf.concat([layer_12,stack_layer_11], -1)
            layer_13 = tf.contrib.layers.fully_connected(stack_layer_12, self.nLRlevel[2], activation_fn=None)
            stack_layer_13 = tf.concat([layer_13,stack_layer_12], -1)
            layer_14 = tf.contrib.layers.fully_connected(stack_layer_13, self.nLRlevel[3], activation_fn=None)
            stack_layer_14 = tf.concat([layer_14,stack_layer_13], -1)
            layer_15 = tf.contrib.layers.fully_connected(stack_layer_14, self.nLRlevel[4], activation_fn=None)
            stack_layer_15 = tf.concat([layer_15,stack_layer_14], -1)

            layer_21 = tf.contrib.layers.fully_connected(stack_layer_12, self.nLRlevel[2], activation_fn=None)
            stack_layer_21 = tf.concat([layer_21, stack_layer_12], -1)
            layer_22 = tf.contrib.layers.fully_connected(stack_layer_21, self.nLRlevel[3], activation_fn=None)
            stack_layer_22 = tf.concat([layer_22, stack_layer_21], -1)

            uniform = tf.contrib.layers.fully_connected(stack_layer_15, self.hsi_size, activation_fn=tf.nn.sigmoid)
            wb = tf.contrib.layers.fully_connected(stack_layer_22, 1, activation_fn=tf.nn.softplus)
            v_samples = self.construct_vsamples(uniform, wb, self.hsi_size)
        return v_samples, uniform, wb

    def encoder_hsi(self, x, dim, reuse=False):
        v_samples,uniform, wb = self.encoder_vsamples_hsi(x, reuse)
        stick_segment_sum_hsi = self.construct_stick_break(v_samples, dim, self.num)
        return stick_segment_sum_hsi

    def encoder_msi(self, x, dim, reuse=False):
        v_samples,v_uniform, v_beta = self.encoder_vsamples_hsi(x, reuse)
        stick_segment_sum_msi = self.construct_stick_break(v_samples, dim, self.num)
        return stick_segment_sum_msi

    def project_hsi(self, x, dim, reuse=False):
        x_input = tf.reshape(x, [-1, self.dim_hsi[-1]])
        proj = tf.matmul(x_input, self.hsi_project['hsi_project_w1'])
        proj_bc = tf.reshape(proj,[dim[0],dim[1],self.dim_msi[-1]])
        # with tf.variable_scope('hsi_project') as scope:
        #     if reuse:
        #         tf.get_variable_scope().reuse_variables()
        # #     proj = tf.layers.dense(x, self.dim_msi[2], activation=None, use_bias=False,name='hsi_project_w1')
        return proj_bc

    def t_mi(self, x, reuse=False):
        h_size = x.get_shape().as_list()
        with tf.variable_scope('t_rmi') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer1 = tf.layers.dense(x, h_size[3], activation=None, use_bias=True)
            layer = tf.layers.dense(layer1, 1, activation=tf.nn.sigmoid, use_bias=False)
        return layer


    def decoder_hsi(self, x, dim, reuse=False):
        x_input = tf.reshape(x,[-1,self.num])
        layer_1 = tf.matmul(x_input, self.whsidecoder['hsi_decoder_w1'])
        layer_2 = tf.matmul(layer_1, self.whsidecoder['hsi_decoder_w2'])
        layer_img = tf.reshape(layer_2,[dim[0],dim[1],self.dim_hsi[-1]])
        return layer_img

    def decoder_msi(self, x, reuse=True):
        layer_1 = self.decoder_hsi(x,self.dim_msi,reuse)
        return layer_1

    def genhsi(self, x, dim, reuse=False):
        encoder_pj = self.project_hsi(x,self.dim_hsi, reuse)
        encoder_op = self.encoder_hsi(encoder_pj, dim, reuse)
        decoder_op = self.decoder_hsi(encoder_op,self.dim_hsi,reuse)
        return decoder_op

    def genmsi(self, x, dim, reuse=False):
        self.hr_reuse = True
        encoder_op = self.encoder_msi(x, dim, reuse)
        decoder_op = self.decoder_msi(encoder_op,reuse)
        decoder_msi = self.project_hsi(decoder_op,self.dim_msi,reuse)
        return decoder_msi

    def genhrhsi(self, x, dim, reuse=True):
        encoder_op = self.encoder_msi(x, dim, reuse)
        decoder_op = self.decoder_msi(encoder_op)
        decoder_hr_hsi = tf.add(decoder_op,self.input.meanhsi)
        return decoder_hr_hsi

    def next_feed(self):
        feed_dict = {self.y_msi:self.input_msi, self.y_hsi:self.input_hsi}
        return feed_dict

    def l1_sparse_loss(self, x):
        y = tf.abs(x)
        sparse_loss = tf.reduce_mean(tf.reduce_sum(y, -1))
        return sparse_loss

    def build_model(self):

        ## build model for lr hsi
        y_pred_hsi = self.genhsi(self.y_hsi, self.dim_hsi)
        y_true_hsi = self.y_hsi
        error_hsi = y_pred_hsi - y_true_hsi
        # hsi_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_hsi, 2),-1),[0,1])
        hsi_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_hsi, 2),[0,1,2]))

        # geometric constraints
        decoder_op = tf.matmul(self.whsidecoder['hsi_decoder_w1'],self.whsidecoder['hsi_decoder_w2'])
        decoder = (decoder_op) + self.input.meanhsi
        hsi_volume_loss = self.l1_sparse_loss(decoder)

        # mutual information
        v_samples_hsi = self.encoder_hsi(self.project_hsi(self.y_hsi,self.dim_hsi,True), self.dim_hsi, reuse=True)
        rep_hsi = tf.expand_dims(v_samples_hsi, 0)
        y = tf.expand_dims(self.project_hsi(self.y_hsi,self.dim_hsi,True),0)
        y_shuffle = tf.random_shuffle(y)
        positive_samples_hsi = tf.concat([y, rep_hsi], -1)
        negative_samples_hsi = tf.concat([y_shuffle, rep_hsi], -1)
        positive_scores_lr = self.t_mi(positive_samples_hsi)
        negative_scores_lr = self.t_mi(negative_samples_hsi,reuse=True)
        hsi_mi_loss = -(tf.reduce_mean(-tf.nn.softplus(-positive_scores_lr))
                          -tf.reduce_mean(tf.nn.softplus(negative_scores_lr)))#-

        # sparse constraint
        eps = 0.00000001
        hsi_top = v_samples_hsi
        hsi_base_norm = tf.reduce_sum(hsi_top, -1, keepdims=True)
        hsi_base_sparse = tf.div(hsi_top, (hsi_base_norm + eps))
        hsi_loss_sparse = tf.reduce_mean(-tf.multiply(hsi_base_sparse, tf.log(hsi_base_sparse + eps)))

        # lr total loss
        hsi_loss = hsi_loss_euc + self.lr_vol_r*hsi_volume_loss \
                   + self.lr_mi_hsi*hsi_mi_loss + self.lr_s_hsi*hsi_loss_sparse


        # for hsi
        theta_encoder = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='hsi_encoder')
        theta_decoder = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='hsi_decoder')
        theta_rmi = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='t_rmi')
        counter_lr = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_hsi = ly.optimize_loss(loss=hsi_loss, learning_rate=self.initlrate,
                                 optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_encoder+theta_decoder+theta_rmi,
                                 global_step=counter_lr)

        # build model for high resolution pan image
        y_pred_msi = self.genmsi(self.y_msi, self.dim_msi, True)
        y_true_msi = self.y_msi
        error_msi = y_pred_msi - y_true_msi
        msi_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_msi, 2),0))

        v_samples_msi=self.encoder_msi(self.y_msi, self.dim_msi, reuse=True)
        rep_msi = tf.expand_dims(v_samples_msi, 0)
        x = tf.expand_dims(self.y_msi, 0)
        x_shuffle = tf.expand_dims(tf.random_shuffle(self.y_msi),0)
        positive_samples_p = tf.concat([x, rep_msi], -1)
        negative_samples_p = tf.concat([x_shuffle, rep_msi], -1)
        positive_scores_p = self.t_mi(positive_samples_p,reuse=True)
        negative_scores_p = self.t_mi(negative_samples_p,reuse=True)
        msi_mi_loss = -(tf.reduce_mean(-tf.nn.softplus(-positive_scores_p))\
                          -tf.reduce_mean(tf.nn.softplus(negative_scores_p)))#-

        # sparse constraint
        msi_top = v_samples_msi
        msi_base_norm = tf.reduce_sum(msi_top, -1, keepdims=True)
        msi_base_sparse = tf.div(msi_top, (msi_base_norm + eps))
        msi_loss_sparse = tf.reduce_mean(-tf.multiply(msi_base_sparse, tf.log(msi_base_sparse + eps)))


        msi_loss = self.initprate*msi_loss_euc + self.lr_mi_hsi*msi_mi_loss \
                   + self.lr_s_hsi * msi_loss_sparse

        counter_p = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_msi = ly.optimize_loss(loss=msi_loss, learning_rate=self.initlrate,
                               optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                               variables= theta_encoder+theta_decoder,
                               global_step=counter_p)

        total_loss  = hsi_loss + msi_loss
        opt_total  = opt_hsi + opt_msi
        return hsi_loss, opt_hsi, msi_loss, opt_msi, hsi_volume_loss, hsi_mi_loss, total_loss, opt_total


    def train(self, load_Path, save_dir, loadLRonly, tol):
        hsi_loss, opt_hsi, msi_loss, opt_msi, hsi_volume_loss, hsi_mi_loss, total_loss, opt_total = self.build_model()

        # record the values
        tf.summary.scalar('volume loss', hsi_volume_loss)
        tf.summary.scalar('lr hsi loss', hsi_mi_loss)
        # tf.summary.scalar('hr msi loss', msi_mi_loss)
        tf.summary.scalar('total hsi loss', hsi_loss)
        tf.summary.scalar('total msi loss', msi_loss)
        merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if os.path.exists(load_Path):
            if loadLRonly:
                # load part of the variables
                vars = tf.contrib.slim.get_variables_to_restore()
                variables_to_restore = [v for v in vars if v.name.startswith('hsi_project/')] \
                                        + [v for v in vars if v.name.startswith('hsi_encoder/')]\
                                        + [v for v in vars if v.name.startswith('hsi_decoder/')]\
                                        + [v for v in vars if v.name.startswith('t_rmi/')]
                saver = tf.train.Saver(variables_to_restore)
                load_file = tf.train.latest_checkpoint(load_Path)
                if load_file==None:
                    print('No checkpoint was saved.')
                else:
                    saver.restore(self.sess,load_file)
            else:
                # load all the variables
                saver = tf.train.Saver(max_to_keep=3)
                load_file = tf.train.latest_checkpoint(load_Path)
                if load_file==None:
                    print('No checkpoint was saved.')
                else:
                    saver.restore(self.sess, load_file)
        else:
            saver = tf.train.Saver(max_to_keep=3)

        results_file_name = pjoin(save_dir,"log_" + "lrate_" + str(self.initlrate)+ ".txt")
        results_file = open(results_file_name, 'a')
        writer = tf.summary.FileWriter(save_dir+'/logs', graph=self.sess.graph)
        feed_dict = self.next_feed()

        saver = tf.train.Saver()
        sam = 10
        rmse_total = zeros(self.epoch+1)
        rmse_total[0] = 1
        dcrate = 0.9994
        for epoch in range(self.epoch):
            _, tloss = self.sess.run([opt_total,total_loss], feed_dict=feed_dict)
            rmse_total[epoch+1] = tloss
            self.initlrate = self.initlrate * dcrate
            self.lr_vol_r = self.lr_vol_r* dcrate
            self.lr_mi_hsi = self.lr_mi_hsi*dcrate
            self.lr_s_hsi = self.lr_s_hsi*dcrate

            if (epoch + 1) % 200 == 0:
                # Report and save progress.
                loss_volume = self.sess.run(hsi_volume_loss, feed_dict=feed_dict)
                volume = "volume of the decoder: {:.12f}"
                volume = volume.format(loss_volume)
                print (volume)
                results_file.write(volume + "\n")
                results_file.flush()

                loss_hsi_mi = self.sess.run(hsi_mi_loss, feed_dict=feed_dict)
                results = "epoch {}: lr en loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, loss_hsi_mi, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

                # loss_msi_mi = self.sess.run(msi_mi_loss, feed_dict=feed_dict)
                # results = "epoch {}: pan en loss {:.12f} learing_rate {:.9f}"
                # results = results.format(epoch, loss_msi_mi, self.initprate)
                # print (results)
                # print ('\n')
                # results_file.write(results + "\n\n")
                # results_file.flush()

                loss_hsi = self.sess.run(hsi_loss, feed_dict=feed_dict)
                results = "epoch {}: hsi loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, loss_hsi, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

                loss_msi = self.sess.run(msi_loss, feed_dict=feed_dict)
                results = "epoch {}: msi loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, loss_msi, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()


                img_lr_hsi = self.sess.run(self.genhsi(self.y_hsi, self.dim_hsi, reuse=True), feed_dict=feed_dict) + self.input.meanhsi
                img_lr_hsi_col = np.reshape(img_lr_hsi,[-1,self.dim_hsi[-1]])
                self.evaluation(img_lr_hsi_col, self.input.colhsi_lr,'LR HSI', epoch, results_file, self.rmse_lr_hsi, self.sam_lr_hsi)

                img_hr_msi = self.sess.run(self.genmsi(self.y_msi, self.dim_msi, reuse=True), feed_dict=feed_dict) + self.input.meanmsi
                img_hr_msi_col = np.reshape(img_hr_msi,[-1,self.dim_msi[-1]])
                self.evaluation(img_hr_msi_col, self.input.colmsi,'HR MSI', epoch, results_file, self.rmse_msi, self.sam_msi)
                #
                # img_hr_hsi = self.sess.run(self.genhrhsi(self.y_msi, self.dim_hrhsi, reuse=True), feed_dict=feed_dict)
                # img_hr_hsi_col = np.reshape(img_hr_hsi,[-1,self.dim_hrhsi[-1]])
                # sam = self.evaluation(img_hr_hsi_col, self.input_hr,'HR HSI', epoch, results_file, self.rmse_msi, self.sam_msi)

                rs = self.sess.run(merged, feed_dict=self.next_feed())
                writer.add_summary(rs, epoch)

            if (epoch+1)%500==0:
                img_lr_hsi = self.sess.run(self.genhsi(self.y_hsi, self.dim_hsi, reuse=True), feed_dict=feed_dict) + self.input.meanhsi
                img_lr_hsi_col = np.reshape(img_lr_hsi,[-1,self.dim_hsi[-1]])
                sam,_= self.evaluation(img_lr_hsi_col, self.input.colhsi_lr,'LR HSI', epoch, results_file, self.rmse_lr_hsi, self.sam_lr_hsi)

                img_hr_msi = self.sess.run(self.genmsi(self.y_msi, self.dim_msi, reuse=True), feed_dict=feed_dict) + self.input.meanmsi
                img_hr_msi_col = np.reshape(img_hr_msi,[-1,self.dim_msi[-1]])
                _, rmse = self.evaluation(img_hr_msi_col, self.input.colmsi,'HR MSI', epoch, results_file, self.rmse_msi, self.sam_msi)

                results_ckpt_name = pjoin(save_dir, "epoch_" + str(epoch)+ "_sam_" + str(round(sam,3)) + "_rmse_" + str(round(rmse,4)) + ".ckpt")
                save_path = saver.save(self.sess, results_ckpt_name)

                results = "weights saved at epoch {}"
                results = results.format(epoch)
                print (results)
                print ('\n')

            elif ((sam < tol)  or (epoch == self.epoch - 1)):
                results_ckpt_name = pjoin(save_dir, "epoch_" + str(epoch) + "_sam_" + str(round(sam,3)) + ".ckpt")
                save_path = saver.save(self.sess, results_ckpt_name)
                img_msi_rp = self.sess.run(self.encoder_hsi(self.y_msi,self.dim_msi, reuse=True),feed_dict=feed_dict)
                img_hr_hsi = self.sess.run(self.genhrhsi(self.y_msi, self.dim_msi, reuse=True), feed_dict=feed_dict)
                check_variable_name = [v.name for v in tf.trainable_variables()]
                proj_name = tf.get_default_graph().get_tensor_by_name(check_variable_name[0])

                proj_weights = self.sess.run(proj_name)
                decoder_w1 = self.sess.run(self.whsidecoder['hsi_decoder_w1'])
                decoder_w2 = self.sess.run(self.whsidecoder['hsi_decoder_w2'])
                result = {'img_hr_hsi':img_hr_hsi,
                          'proj_weights':proj_weights,
                          'img_msi_rp':img_msi_rp,
                          'decoder_w1':decoder_w1,
                          'decoder_w2':decoder_w2}
                sio.savemat(save_dir + "/hr_hsi_out.mat", result)
                break;

        return save_path

    def evaluation(self,img_hr,img_tar,name,epoch,results_file,rmse_list,sam_list):
        # evalute the results
        # ref = img_tar*255.0
        # tar = img_hr*255.0
        # lr_flags = tar<0
        # tar[lr_flags]=0
        # hr_flags = tar>255.0
        # tar[hr_flags] = 255.0
        ref = img_tar
        tar = img_hr
        lr_flags = tar<0
        tar[lr_flags]=0
        hr_flags = tar>1
        tar[hr_flags] = 1

        #ref = ref.astype(np.int)
        #tar = tar.astype(np.int)

        diff = ref - tar;
        size = ref.shape
        rmse = np.sqrt( np.sum(np.sum(np.power(diff,2))) / (size[0]*size[1]));
        rmse_list.append(rmse)
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
        sam_list.append(sam)
        # print('epoch '+str(epoch)+' '+'The SAM of the ' + name + ' is: '+ str(sam)+'\n')

        results = name + " epoch {}: SAM  {:.12f} "
        results = results.format(epoch,  sam)
        print (results)
        print ("\n")
        results_file.write(results + "\n")
        results_file.flush()
        return sam,rmse

    def testhr(self,save_dir,load_path):
        # gen_lr = self.genLR(self.y_lr)
        # gen_Pan = self.genPan(self.y_lri)
        # gen_hr  = self.genHR(self.x_pan,reuse=False)
        # gen_hidden_lri  = self.encoder_Pan(self.y_lri,reuse=False)
        # gen_lr_v, gen_lr_uniform, gen_lr_beta = self.encoder_vsamples_lr(self.y_lr);
        # gen_hr_v, gen_hr_uniform, gen_hr_beta = self.encoder_vsamples_lr(self.x_pan);

        print(load_path)
        lrhsi = self.genhsi(self.y_hsi, self.dim_hsi, reuse=False)
        load_file = tf.train.latest_checkpoint(load_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, load_file)
        feed_dict = self.next_feed()

        img_msi_rp = self.sess.run(self.encoder_hsi(self.y_msi, self.dim_msi, reuse=True), feed_dict=feed_dict)
        img_hr_hsi = self.sess.run(self.genhrhsi(self.y_msi, self.dim_msi, reuse=True), feed_dict=feed_dict)
        check_variable_name = [v.name for v in tf.trainable_variables()]
        proj_name = tf.get_default_graph().get_tensor_by_name(check_variable_name[0])

        proj_weights = self.sess.run(proj_name)
        decoder_w1 = self.sess.run(self.whsidecoder['hsi_decoder_w1'])
        decoder_w2 = self.sess.run(self.whsidecoder['hsi_decoder_w2'])
        # decoder_w3 = self.sess.run(self.whsidecoder['hsi_decoder_w3'])

        # img_hr_msi = self.sess.run(self.savepan(self.x_pan, reuse=True), feed_dict=feed_dict)
        result = {'img_hr_hsi': img_hr_hsi,
                  'proj_weights': proj_weights,
                  'img_msi_rp': img_msi_rp,
                  'decoder_w1': decoder_w1,
                  'decoder_w2': decoder_w2}
        sio.savemat(save_dir + "/hr_hsi_out.mat", result)




