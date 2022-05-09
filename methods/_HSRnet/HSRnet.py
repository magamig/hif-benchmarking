
"""
# This is the training code for the HSRnet (CAVE)
# Hyperspectral Image Super-resolution via Deep Spatio-spectral Attention Convolutional Neural Networks
# author: Jin-Fan Hu
"""

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.layers as ly
import os
import h5py
import scipy.io as sio
import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _phase_shift(I, r):
    bsize, w, h, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]
    X = tf.reshape(I, (bsize, w, h, r, r))
    X = tf.split(X, w, 1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
    X = tf.split(X, h, 1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)

    return tf.reshape(X, (bsize, w * r, h * r, 1))


def PS(X, r):
    Xc = tf.split(X, 31, 3)
    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    return X


def vis_ms(data):
    _, b, g, _, r, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = tf.split(data, 31,
                                                                                                           axis=3)
    vis = tf.concat([r, g, b], axis=3)
    return vis


# rgbNet structures
def rgbNet(ms, RGB, num_spectral=31, num_res=6, num_fm=64, reuse=False):
    weight_decay = 1e-4

    with tf.variable_scope('net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ## Channel Attention
        gap_ms_c = tf.reduce_mean(ms, [1, 2], name='global_pool', keep_dims=True)

        with tf.compat.v1.variable_scope('CA'):
            CA = ly.conv2d(gap_ms_c, num_outputs=1, kernel_size=1, stride=1,
                           weights_regularizer=ly.l2_regularizer(weight_decay),
                           weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
            CA = ly.conv2d(CA, num_outputs=num_spectral, kernel_size=1, stride=1,
                           weights_regularizer=ly.l2_regularizer(weight_decay),
                           weights_initializer=tf.random_normal_initializer(), activation_fn=tf.nn.sigmoid)

        ## Spatial Attention
        gap_RGB_s = tf.reduce_mean(RGB, [3], name='global_pool', keep_dims=True)

        SA = ly.conv2d(gap_RGB_s, num_outputs=1, kernel_size=6, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.sigmoid)

        sa = ly.conv2d(SA, 1, 6, 4, activation_fn=tf.nn.sigmoid,
                       weights_initializer=ly.variance_scaling_initializer(),
                       weights_regularizer=ly.l2_regularizer(weight_decay))
        ## downsampled RGB
        rgb = ly.conv2d(RGB, 3, 6, 4, activation_fn=None,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
        rslice, gslice, bslice = tf.split(rgb, 3, axis=3)
        msp1, msp2 = tf.split(ms, [15, 16], axis=3)
        ms = tf.concat([rslice, msp1, gslice, msp2, bslice], axis=3)

        rs = ly.conv2d(ms, num_outputs=num_spectral * 4 * 4, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        rs = PS(rs, 4)

        Rslice, Gslice, Bslice = tf.split(RGB, 3, axis=3)
        Msp1, Msp2 = tf.split(rs, [15, 16], axis=3)
        rs = tf.concat([Rslice, Msp1, Gslice, Msp2, Bslice], axis=3)
        rs = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        ## ResNet Blocks
        for i in range( num_res):
            rs1 = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
            rs1 = ly.conv2d(rs1, num_outputs=num_fm, kernel_size=3, stride=1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
            rs = tf.add(rs, rs1)
        
        rs = SA * rs
        rs = ly.conv2d(rs, num_outputs=num_spectral, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
        rs = CA * rs

        return rs

def train():

    tf.reset_default_graph()

    train_batch_size = 32  # training batch size
    test_batch_size = 32  # validation batch size
    image_size = 64 # patch size
    bands = 31
    iterations = 150001  # total number of iterations to use. 
    train_data_name = 'train4(20-11)(pRGB).mat'  # training data (v7.3 mat)
    test_data_name = 'validation4(20-11)(pRGB).mat'  # validation data (v7.3 mat)
    restore = False  # load existing model or not
    method = 'Adam'  # training method: Adam or SGD


    train_data = h5py.File(train_data_name)  # for large data ( v7.3 data)

    test_data = h5py.File(test_data_name)

    ############## placeholder for training
    gt = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size, image_size, bands])
    lms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size, image_size, bands])
    ms_hp = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size // 4, image_size // 4, bands])
    rgb_hp = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size, image_size, 3])

    ############# placeholder for testing
    test_gt = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, bands])
    test_lms = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, bands])
    test_ms_hp = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size // 4, image_size // 4, bands])
    test_rgb_hp = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, 3])


    mrs,_,_ = rgbNet(ms_hp, rgb_hp)
    mrs = tf.add(mrs, lms)

    test_rs,_,_ = rgbNet(test_ms_hp, test_rgb_hp, reuse=True)
    test_rs = test_rs + test_lms

    ######## loss function
    ##### L1
    mse = tf.reduce_mean(tf.abs(mrs - gt))
    test_mse = tf.reduce_mean(tf.abs(test_rs - test_gt))
    ##### L2
    # mse = tf.reduce_mean(tf.square(mrs - gt))
    # test_mse = tf.reduce_mean(tf.square(test_rs - test_gt))

    ##### Loss summary
    mse_loss_sum = tf.summary.scalar("mse_loss", mse)

    test_mse_sum = tf.summary.scalar("test_loss", test_mse)

    lms_sum = tf.summary.image("lms", tf.clip_by_value(vis_ms(lms), 0, 1))
    mrs_sum = tf.summary.image("rs", tf.clip_by_value(vis_ms(mrs), 0, 1))

    label_sum = tf.summary.image("label", tf.clip_by_value(vis_ms(gt), 0, 1))

    all_sum = tf.summary.merge([mse_loss_sum, mrs_sum, label_sum, lms_sum])

    #########   optimal    Adam or SGD

    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')

    if method == 'Adam':
        g_optim = tf.train.AdamOptimizer(0.0001, beta1=0.9) \
            .minimize(mse, var_list=t_vars)

    else:
        global_steps = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.1, global_steps, decay_steps=50000, decay_rate=0.1)
        clip_value = 0.1 / lr
        optim = tf.train.MomentumOptimizer(lr, 0.9)
        gradient, var = zip(*optim.compute_gradients(mse, var_list=t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient, clip_value)
        g_optim = optim.apply_gradients(zip(gradient, var), global_step=global_steps)

    ##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #### Run the above
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(init)

        if restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess, ckpt.model_checkpoint_path)

        #### read training data #####
        gt1 = train_data['gt'][...]  ## GT N*H*W*C
        rgb1 = train_data['rgb1'][...]  #### HR-MSI N*H*W*c
        ms_lr1 = train_data['ms'][...]  ### LR-HSI N*h*w*C
        lms1 = train_data['lms'][...]  #### Upsampled LR-HSI

        gt1 = np.array(gt1, dtype=np.float32) / (2 ** 16 - 1)  ### normalization
        rgb1 = np.array(rgb1, dtype=np.float32) / (2 ** 8 - 1)
        ms_lr1 = np.array(ms_lr1, dtype=np.float32) / (2 ** 16 - 1)
        lms1 = np.array(lms1, dtype=np.float32) / (2 ** 16 - 1)

        N = gt1.shape[0]

        #### read validation data #####
        gt2 = test_data['gt'][...]  ## GT N*H*W*C
        rgb2 = test_data['rgb1'][...]  #### HR-MSI N*H*W*c
        ms_lr2 = test_data['ms'][...]  ### LR-HSI N*h*w*C
        lms2 = test_data['lms'][...]  #### Upsampled LR-HSI

        gt2 = np.array(gt2, dtype=np.float32) / (2 ** 16 - 1)
        rgb2 = np.array(rgb2, dtype=np.float32) / (2 ** 8 - 1)
        ms_lr2 = np.array(ms_lr2, dtype=np.float32) / (2 ** 16 - 1)
        lms2 = np.array(lms2, dtype=np.float32) / (2 ** 16 - 1)
        N2 = gt2.shape[0]

        mse_train = []
        mse_valid = []

        for i in range(iterations):
            ###################################################################
            #### training phase! ###########################

            bs = train_batch_size
            batch_index = np.random.randint(0, N, size=bs)

            train_gt = gt1[batch_index, :, :, :]
            rgb_batch = rgb1[batch_index, :, :, :]
            ms_lr_batch = ms_lr1[batch_index, :, :, :]
            train_lms = lms1[batch_index, :, :, :]


            _, mse_loss, merged = sess.run([g_optim, mse, all_sum], feed_dict={gt: train_gt, lms: train_lms,
                                                                               ms_hp: ms_lr_batch,
                                                                               rgb_hp: rgb_batch })

            if i % 1000 == 0:
                mse_train.append(mse_loss)  # record the loss of trainning
                print("Iter: " + str(i) + " loss: " + str(mse_loss))  # print, e.g.,: Iter: 0 loss: 0.18406609


            if i % 10000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/model-' + str(i) + '.ckpt')
                print("Save Model")

            ###################################################################
            #### compute the loss of validation data ###########################
            bs_test = test_batch_size
            batch_index2 = np.random.randint(0, N2, size=bs_test)

            if i % 2000 == 0 and i != 0:
                test_gt_batch = gt2[batch_index2, :, :, :]
                test_rgb_batch = rgb2[batch_index2, :, :, :]
                test_ms_lr_batch = ms_lr2[batch_index2, :, :, :]
                test_lms_batch = lms2[batch_index2, :, :, :]

                test_mse_loss, merged = sess.run([test_mse, test_mse_sum],
                                                 feed_dict={test_gt: test_gt_batch, test_lms: test_lms_batch,
                                                            test_ms_hp: test_ms_lr_batch,
                                                            test_rgb_hp: test_rgb_batch})
                mse_valid.append(test_mse_loss)  # record the loss of validation
                print("Iter: " + str(i) + " Valid loss: " + str(test_mse_loss))


def test():
    test_data = 'test_cave_demo.mat'
    tf.reset_default_graph()
    N = 1
    sz = 512
    OUT = np.zeros((N, sz, sz, 31))
    r_hp = tf.placeholder(shape=[1, sz, sz, 3], dtype=tf.float32)
    m_hp = tf.placeholder(shape=[1, sz // 4, sz // 4, 31], dtype=tf.float32)
    lms_p = tf.placeholder(shape=[1, sz, sz, 31], dtype=tf.float32)

    rs = rgbNet(m_hp, r_hp)  # output high-frequency parts

    mrs = tf.add(rs, lms_p)

    output = tf.clip_by_value(mrs, 0, 1)  # final output



    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(init)

        # loading  model
        ckpt = tf.train.latest_checkpoint(model_directory)
        saver.restore(sess, ckpt)
        print("load new model")
        # data = h5py.File(test_data) ## for v7.3 mat
        data = sio.loadmat(test_data)
        Inms = data['ms']
        Inlms = data['lms'][...]
        Inrgb = data['rgb1'][...]
        for i in range(N):


            ms = Inms[i, :, :, :] / (2 ** 16 - 1)
            ms = ms[np.newaxis,:,:,:]

            lms = Inlms[i, :, :, :] / (2 ** 16 - 1)
            lms = lms[np.newaxis,:,:,:]

            rgb = Inrgb[i, :, :, :] / (2 ** 8 - 1)
            rgb = rgb[np.newaxis,:,:,:]

            ms_in = ms
            rgb_in = rgb

            [final_output] = sess.run([output],feed_dict={r_hp: rgb_in, m_hp: ms_in, lms_p: lms})
            OUT[i, :, :, :] = final_output

            print('testing image' + str(i+1) + ' is done!')
        sio.savemat('output-HSRnet-cave.mat',
            {'output': OUT})

if __name__ == '__main__':
    global model_directory
    model_directory = 'models(cave)'
    # train()
    test()







