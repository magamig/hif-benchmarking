# Patch-Aware Deep Hyperspectral and Multispectral Image Fusion by Unfolding Subspace-Based Optimization Model
# Jianjun Liu, JSTARS, 2022
import tensorflow as tf
import tensorly as tl
import numpy as np
import scipy.io as sio
import os
import scipy
import time
import shutil

from tensorflow.contrib import layers

from SpfNet.utils import FusionNet
from SpfNet.funs import check_dir


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1)


class SpfNet(FusionNet):
    def __init__(self, data_num, sim=True):
        super().__init__(data_num, sim)
        # one can redefine param here
        self.k = min(self.hs_bands, 31)
        self.restart_epoch = 2
        # train and valid, SVD
        self.train_num, self.valid_num = self.train_valid_process_piece(self.sim_save_path, block=self.block)
        # variable, graph, loss and etc.
        self.__X = tf.placeholder(tf.float32, shape=(None, None, None, self.hs_bands), name='HRHS')
        self.__Y = tf.placeholder(tf.float32, shape=(None, None, None, self.hs_bands), name='LRHS')
        self.__Z = tf.placeholder(tf.float32, shape=(None, None, None, self.ms_bands), name='HRMS')
        self.__A = tf.placeholder(tf.float32, shape=(None, self.hs_bands, self.k), name='A')
        self.__lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.__outX = self.build_graph(self.__Y, self.__Z, self.__A)
        self.__psnr, self.__ssim = self.build_psnr_ssim(self.__X, self.__outX)
        self.__loss = self.build_loss(self.__X, self.__outX, norm='l1')
        self.train_op = tf.train.AdamOptimizer(self.__lr).minimize(self.__loss)
        self.saver = tf.train.Saver(max_to_keep=20)
        self.decrease_lr = self.decrease_lr_v2
        if data_num == 2 or data_num == 11:
            self.decrease_lr = self.decrease_lr_v0

    def build_graph(self, Y, Z, A):
        out_stages = 5
        in_steps = 3
        XList = self.spat_fusion_net('spat_fusion', Y, Z, A, out_stages=out_stages, in_steps=in_steps)
        X = tf.add_n(XList)
        X = tf.scalar_mul(1 / out_stages, X)
        return X

    def spat_fusion_net(self, name, Y, Z, A, out_stages=5, in_steps=3):
        with tf.variable_scope(name):
            AtA = tf.einsum('NBk,NBl->Nkl', A, A)  # remove ???
            AtY = tf.einsum('NBk,NwhB->Nwhk', A, Y)
            S = tf.zeros(shape=[tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[2], self.k], dtype=tf.float32, name='S0')
            V, D = S, S
            V1, D1 = S, S
            XList = []
            for out_iter in range(out_stages):
                R = tf.get_variable('R%s' % out_iter, [self.ms_bands, self.hs_bands], tf.float32,
                                    initializer=tf.constant_initializer(1 / self.hs_bands))
                RA = tf.einsum('bB,NBk->Nbk', R, A)
                RAtRA = tf.einsum('Nbk,Nbl->Nkl', RA, RA)
                RAtZ = tf.einsum('Nbk,NWHb->NWHk', RA, Z)  # move in subnet ???
                for in_iter in range(in_steps):
                    S = self.spat_fusion_subnet('fus_subnet',
                                                S, AtA, AtY, RAtRA, RAtZ, V, D, V1, D1, out_iter, in_iter)
                V = self.down_res_cnn_module('down%s' % out_iter, S - D)
                D = D - (S - V)
                V1 = self.up_res_cnn_module('up%s' % out_iter, S - D1)
                D1 = D1 - (S - V1)
                X = tf.einsum('NBk,NWHk->NWHB', A, S)
                X = layers.conv2d(X, self.hs_bands, 3, 1, activation_fn=None,
                                  weights_initializer=layers.variance_scaling_initializer(),
                                  weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                  scope="out" + str(out_iter))
                XList.append(X)
            return XList

    def spat_fusion_subnet(self, name, S, AtA, AtY, RAtRA, RAtZ, V, D, V1, D1, out_iter, in_iter):
        stride, ker_size = self.ratio, self.ratio+self.ratio//2
        with tf.variable_scope(name + '_' + str(in_iter), reuse=tf.AUTO_REUSE):
            S1 = tf.einsum('Nkl,NWHk->NWHl', AtA, S)
            S1 = layers.conv2d(S1, self.k, ker_size, stride, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope='B')
            S1 = S1 - AtY
            S1 = layers.conv2d_transpose(S1, self.k, ker_size, stride, activation_fn=None,
                                         weights_initializer=layers.variance_scaling_initializer(),
                                         weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                         scope='Bt')
            S2 = tf.einsum('Nkl,NWHk->NWHl', RAtRA, S)
            S2 = S2 - RAtZ
        with tf.variable_scope(name + '_' + str(out_iter) + '_' + str(in_iter)):
            S3 = tf.concat([S1, S2, S - V - D, S - V1 - D1], axis=-1)
            S3 = layers.conv2d(S3, self.k, 3, 1, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope="fus")
            S = S - S3
            return S

    def up_res_cnn_module(self, name, input_tensor, filter_size=3):
        channel = input_tensor.shape[-1]
        with tf.variable_scope(name):
            X1 = layers.conv2d_transpose(input_tensor, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                                         weights_initializer=layers.variance_scaling_initializer(),
                                         weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                         scope='u1')
            Y1 = layers.conv2d(X1, channel, filter_size, 1, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope='cnn')
            Y1 = Y1 + X1
            output = layers.conv2d(Y1, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                                   weights_initializer=layers.variance_scaling_initializer(),
                                   weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                   scope='d1')
            return tf.add(output, input_tensor)

    def down_res_cnn_module(self, name, input_tensor, filter_size=3):
        channel = input_tensor.shape[-1]
        with tf.variable_scope(name):
            X1 = layers.conv2d(input_tensor, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope='d1')
            X2 = layers.conv2d(X1, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope='d2')
            X3 = layers.conv2d(X2, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope='d3')
            Y3 = layers.conv2d(X3, channel, filter_size, 1, activation_fn=tf.nn.leaky_relu,
                               weights_initializer=layers.variance_scaling_initializer(),
                               weights_regularizer=layers.l2_regularizer(self.l2_decay),
                               scope='cnn')
            Y3 = Y3 + X3
            Y2 = layers.conv2d_transpose(Y3, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                                         weights_initializer=layers.variance_scaling_initializer(),
                                         weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                         scope='u3')
            Y2 = Y2 + X2
            Y1 = layers.conv2d_transpose(Y2, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                                         weights_initializer=layers.variance_scaling_initializer(),
                                         weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                         scope='u2')
            Y1 = Y1 + X1
            output = layers.conv2d_transpose(Y1, channel, filter_size, 2, activation_fn=tf.nn.leaky_relu,
                                             weights_initializer=layers.variance_scaling_initializer(),
                                             weights_regularizer=layers.l2_regularizer(self.l2_decay),
                                             scope='u1')
            return tf.add(output, input_tensor)

    # training using tfrecords files
    def train_tf(self, restart=False):
        with self.init_device() as sess:
            if restart is False:
                sess.run(tf.global_variables_initializer())
            else:
                latest_model = tf.train.get_checkpoint_state(self.model_save_path)
                self.saver.restore(sess, latest_model.model_checkpoint_path)
                self.current_epoch = 0
            count = 0
            for self.current_restart in range(self.restart_epoch):
                _opt = self.train_op
                _loss = self.__loss
                train_batch_X, train_batch_Y, train_batch_Z, train_batch_A = self.train_data_tf()
                try:
                    start = time.perf_counter()
                    while True:
                        train_X, train_Y, train_Z, train_A = sess.run(
                            [train_batch_X, train_batch_Y, train_batch_Z, train_batch_A])
                        _, loss = sess.run([_opt, _loss], feed_dict={self.__X: train_X,
                                                                     self.__Y: train_Y,
                                                                     self.__Z: train_Z,
                                                                     self.__A: train_A[:, :, 0:self.k],
                                                                     self.__lr: self.lr})
                        count += train_X.shape[0]
                        if count // self.train_num > self.current_epoch:
                            end = time.perf_counter()
                            print('training', end - start)
                            start1 = time.perf_counter()
                            self.current_epoch += 1
                            valid_ls = self.valid_tf(sess)
                            self.decrease_lr()
                            end1 = time.perf_counter()
                            print('validation', end1 - start1)
                            start = time.perf_counter()
                except tf.errors.OutOfRangeError:
                    print("Finish train_tf")
            pass

    # validation using tfrecords files
    def valid_tf(self, sess):
        _psnr = self.__psnr
        _ssim = self.__ssim
        _loss = self.__loss
        valid_batch_X, valid_batch_Y, valid_batch_Z, valid_batch_A = self.valid_data_tf()
        psnr, ssim, loss = 0, 0, 0
        try:
            while True:
                valid_X, valid_Y, valid_Z, valid_A = sess.run(
                    [valid_batch_X, valid_batch_Y, valid_batch_Z, valid_batch_A])
                psnr_val, ssim_val, loss_val = sess.run([_psnr, _ssim, _loss],
                                                        feed_dict={self.__X: valid_X,
                                                                   self.__Y: valid_Y,
                                                                   self.__Z: valid_Z,
                                                                   self.__A: valid_A[:, :, 0:self.k]})
                psnr += psnr_val
                ssim += ssim_val
                loss += loss_val
        except tf.errors.OutOfRangeError:
            print("validation end")
        psnr = psnr / self.valid_num
        ssim = ssim / self.valid_num
        print('----valid-----epoch: %s, psnr:%s  ssim:%s  loss:%s  lr:%s-----'
              % (self.current_epoch, psnr, ssim, loss, self.lr))
        t = psnr * 0.5 + ssim * 0.5
        if t > self.max_power:
            print('get a satisfying model')
            self.max_power = t
            self.saver.save(sess, self.model_save_path, global_step=self.current_epoch)
        return loss

    # test by feeding the images entirely
    def test(self):
        run_time = 0
        with self.init_device() as sess:
            latest_model = tf.train.get_checkpoint_state(self.model_save_path)
            self.saver.restore(sess, latest_model.model_checkpoint_path)
            for i in range(self.test_start, self.test_end + 1):
                mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
                start = time.perf_counter()
                tY = mat['LRHS']
                tZ = mat['HRMS']
                tA, _, _ = scipy.linalg.svd(tl.unfold(tY, mode=2), full_matrices=False)
                tA = tA[:, 0:self.k]
                tY = np.expand_dims(tY, 0)
                tZ = np.expand_dims(tZ, 0)
                tA = np.expand_dims(tA, 0)
                output = sess.run(self.__outX, feed_dict={self.__Y: tY, self.__Z: tZ, self.__A: tA})
                output = np.squeeze(output)
                # output[output < 0] = 0.0
                # output[output > 1] = 1.0
                end = time.perf_counter()
                run_time += end - start
                check_dir(self.output_save_path)
                sio.savemat(self.output_save_path + '%d.mat' % i, {'F': output, 'TM': end - start})
                print('test: %d has finished' % i)
        print('Time: %ss' % (run_time / (self.test_end - self.test_start + 1)))

    # test by feeding the images piece by piece
    def test_piece(self, stride=None):
        if stride is None:
            stride = self.test_stride
        run_time = 0
        with self.init_device() as sess:
            latest_model = tf.train.get_checkpoint_state(self.model_save_path)
            self.saver.restore(sess, latest_model.model_checkpoint_path)
            for i in range(self.test_start, self.test_end + 1):
                mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
                tY = mat['LRHS']
                tZ = mat['HRMS']
                output = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
                num_sum = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
                start = time.perf_counter()
                for x in range(0, tZ.shape[0] - self.piece_size + 1, stride):
                    for y in range(0, tZ.shape[1] - self.piece_size + 1, stride):
                        end_x = x + self.piece_size
                        if end_x + stride > tZ.shape[0]:
                            end_x = tZ.shape[0]
                        end_y = y + self.piece_size
                        if end_y + stride > tZ.shape[1]:
                            end_y = tZ.shape[1]
                        itY = tY[x // self.ratio:end_x // self.ratio, y // self.ratio:end_y // self.ratio, :]
                        itZ = tZ[x:end_x, y:end_y, :]
                        itA, _, _ = scipy.linalg.svd(tl.unfold(itY, mode=2), full_matrices=False)
                        itA = itA[:, 0:self.k]
                        itY = np.expand_dims(itY, 0)
                        itZ = np.expand_dims(itZ, 0)
                        itA = np.expand_dims(itA, 0)
                        tmp = sess.run(self.__outX, feed_dict={self.__Y: itY, self.__Z: itZ, self.__A: itA})
                        tmp = np.squeeze(tmp)
                        output[x:end_x, y:end_y, :] += tmp
                        num_sum[x:end_x, y:end_y, :] += 1
                output = output / num_sum
                # output[output < 0] = 0.0
                # output[output > 1] = 1.0
                end = time.perf_counter()
                run_time += end - start
                check_dir(self.output_save_path)
                sio.savemat(self.output_save_path + '%d.mat' % i, {'F': output, 'TM': end - start})
                print('test: %d has finished' % i)
        print('Time: %ss' % (run_time / (self.test_end - self.test_start + 1)))

    # test by feeding the images piece by piece, pieces are stored in tfrecords files
    def test_piece_tf(self, stride=None, recompute=True):
        if stride is None:
            stride = self.test_stride
        # cut piece and save to tf files
        if recompute is True:
            self.test_process_piece(self.sim_save_path, stride)
        run_time = 0
        with self.init_device() as sess:
            latest_model = tf.train.get_checkpoint_state(self.model_save_path)
            self.saver.restore(sess, latest_model.model_checkpoint_path)
            for i in range(self.test_start, self.test_end + 1):
                mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
                tY = mat['LRHS']
                tZ = mat['HRMS']
                output = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
                num_sum = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
                file_list = self.test_data_path + str(i) + '.tfrecords'
                test_batch_Y, test_batch_Z, test_batch_A, test_batch_pos = self.test_data_tf(file_list)
                start = time.perf_counter()
                try:
                    while True:
                        test_Y, test_Z, test_A, test_pos = sess.run(
                            [test_batch_Y, test_batch_Z, test_batch_A, test_batch_pos])
                        X = sess.run(self.__outX, feed_dict={self.__Y: test_Y,
                                                             self.__Z: test_Z,
                                                             self.__A: test_A[:, :, 0:self.k]})
                        test_pos = np.squeeze(test_pos, axis=1)
                        test_pos = test_pos.astype(np.int)
                        for j in range(X.shape[0]):
                            x, end_x, y, end_y = test_pos[j, 0], test_pos[j, 1], test_pos[j, 2], test_pos[j, 3]
                            output[x:end_x, y:end_y, :] += X[j, :, :, :]
                            num_sum[x:end_x, y:end_y, :] += 1
                except tf.errors.OutOfRangeError:
                    print("test%d end" % i)
                output = output / num_sum
                # output[output < 0] = 0.0
                # output[output > 1] = 1.0
                end = time.perf_counter()
                run_time += end - start
                check_dir(self.output_save_path)
                sio.savemat(self.output_save_path + '%d.mat' % i, {'F': output, 'TM': end - start})
                print('test: %d has finished' % i)
        print('Time: %ss' % (run_time / (self.test_end - self.test_start + 1)))

    @staticmethod
    def _parse_record(example):
        features = {'HS_shape': tf.FixedLenFeature([3], tf.int64),
                    'LRHS_shape': tf.FixedLenFeature([3], tf.int64),
                    'HRMS_shape': tf.FixedLenFeature([3], tf.int64),
                    'A_shape': tf.FixedLenFeature([2], tf.int64),
                    'HS': tf.FixedLenFeature([], tf.string),
                    'LRHS': tf.FixedLenFeature([], tf.string),
                    'HRMS': tf.FixedLenFeature([], tf.string),
                    'A': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example, features)
        HS_shape = parsed_features['HS_shape']
        HS = tf.decode_raw(parsed_features['HS'], tf.float32)
        HS = tf.reshape(HS, HS_shape)
        LRHS_shape = parsed_features['LRHS_shape']
        LRHS = tf.decode_raw(parsed_features['LRHS'], tf.float32)
        LRHS = tf.reshape(LRHS, LRHS_shape)
        HRMS_shape = parsed_features['HRMS_shape']
        HRMS = tf.decode_raw(parsed_features['HRMS'], tf.float32)
        HRMS = tf.reshape(HRMS, HRMS_shape)
        A_shape = parsed_features['A_shape']
        A = tf.decode_raw(parsed_features['A'], tf.float32)
        A = tf.reshape(A, A_shape)
        return HS, LRHS, HRMS, A

    # -----------------------------below is related processing -----------------------------------
    def train_valid_process_piece(self, sim_path, block, valid_ratio=0.2):  # HS/HRMS/LRHS/A
        if os.path.exists(self.valid_data_path + '1.tfrecords'):
            print('Train and Validation: xx exists!')
            num = sio.loadmat(self.train_data_path + 'info.mat')
            train_num = num['num']
            num = sio.loadmat(self.valid_data_path + 'info.mat')
            valid_num = num['num']
            return train_num, valid_num
        elif os.path.exists(self.valid_data_path + 'info.mat'):
            print('Train and Validation: xx piece exists!')
            num = sio.loadmat(self.train_data_path + 'info.mat')
            train_num = num['num']
            num = sio.loadmat(self.valid_data_path + 'info.mat')
            valid_num = num['num']
            self.train_valid_piece_tf(self.train_data_path, self.valid_data_path)
            return train_num, valid_num
        # the first total_img is processed
        check_dir(self.train_data_path)
        check_dir(self.valid_data_path)
        train_num, valid_num = 0, 0
        for i in range(self.train_start, self.train_end + 1):
            mat = sio.loadmat(sim_path + '%d.mat' % i)
            X = mat['HS']
            Y = mat['LRHS']
            Z = mat['HRMS']
            rows, cols, _ = X.shape
            for x in range(0, rows - self.piece_size + 1, self.stride):
                for y in range(0, cols - self.piece_size + 1, self.stride):
                    label = X[x:x + self.piece_size, y:y + self.piece_size, :]
                    z_data = Z[x:x + self.piece_size, y:y + self.piece_size, :]
                    y_data = Y[x // self.ratio:(x + self.piece_size) // self.ratio,
                             y // self.ratio:(y + self.piece_size) // self.ratio, :]
                    A, _, _ = scipy.linalg.svd(tl.unfold(y_data, mode=2), full_matrices=False)
                    mat_dict = {'HS': label, 'LRHS': y_data, 'HRMS': z_data, 'A': A}
                    train_num, valid_num = self.train_valid_piece_save(train_num, valid_num, mat_dict,
                                                                       self.train_data_path, self.valid_data_path,
                                                                       block, valid_ratio)
            print('Piece: %d has finished' % i)
        print('Piece done')
        # --------------------- save information for further usage: info.mat -------------------------------
        sio.savemat(self.train_data_path + 'info.mat', {'num': train_num})
        sio.savemat(self.valid_data_path + 'info.mat', {'num': valid_num})
        self.train_valid_piece_tf(self.train_data_path, self.valid_data_path)
        return train_num, valid_num

    def test_process_piece(self, sim_path, stride):  # HRMS/LRHS/A/pos
        # shutil.rmtree(self.test_data_path)
        # the test image is processed
        check_dir(self.test_data_path)
        for i in range(self.test_start, self.test_end + 1):
            count = 0
            mat = sio.loadmat(sim_path + '%d.mat' % i)
            tY = mat['LRHS']
            tZ = mat['HRMS']
            for x in range(0, tZ.shape[0] - self.piece_size + 1, stride):
                for y in range(0, tZ.shape[1] - self.piece_size + 1, stride):
                    end_x = x + self.piece_size
                    if end_x + stride > tZ.shape[0]:
                        end_x = tZ.shape[0]
                    end_y = y + self.piece_size
                    if end_y + stride > tZ.shape[1]:
                        end_y = tZ.shape[1]
                    itY = tY[x // self.ratio:end_x // self.ratio, y // self.ratio:end_y // self.ratio, :]
                    itZ = tZ[x:end_x, y:end_y, :]
                    itA, _, _ = scipy.linalg.svd(tl.unfold(itY, mode=2), full_matrices=False)
                    pos = np.array([x, end_x, y, end_y], ndmin=2)
                    mat_dict = {'LRHS': itY, 'HRMS': itZ, 'A': itA, 'pos': pos}
                    count += 1
                    check_dir(self.test_data_path + str(i) + '/')
                    sio.savemat(self.test_data_path + str(i) + '/%d.mat' % count, mat_dict)
            print('Piece: %d has finished' % i, 'total: %d' % count)
        print('Piece done')
        folders = os.listdir(self.test_data_path)
        for dir_name in folders:
            input_path = self.test_data_path + dir_name + '/'
            output_file = self.test_data_path + dir_name + '.tfrecords'
            if os.path.isdir(input_path):
                self.files_to_tf(input_path, output_file)
                print(input_path, output_file)
        return

    def test_data_tf(self, file_list):
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(self.parse_record, num_parallel_calls=None)
        dataset = dataset.batch(batch_size=self.valid_batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    @staticmethod
    def parse_record(example):  # LRHS/HRMS/A/pos
        features = {'pos_shape': tf.FixedLenFeature([2], tf.int64),
                    'LRHS_shape': tf.FixedLenFeature([3], tf.int64),
                    'HRMS_shape': tf.FixedLenFeature([3], tf.int64),
                    'A_shape': tf.FixedLenFeature([2], tf.int64),
                    'pos': tf.FixedLenFeature([], tf.string),
                    'LRHS': tf.FixedLenFeature([], tf.string),
                    'HRMS': tf.FixedLenFeature([], tf.string),
                    'A': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example, features)
        pos_shape = parsed_features['pos_shape']
        pos = tf.decode_raw(parsed_features['pos'], tf.float32)
        pos = tf.reshape(pos, pos_shape)
        LRHS_shape = parsed_features['LRHS_shape']
        LRHS = tf.decode_raw(parsed_features['LRHS'], tf.float32)
        LRHS = tf.reshape(LRHS, LRHS_shape)
        HRMS_shape = parsed_features['HRMS_shape']
        HRMS = tf.decode_raw(parsed_features['HRMS'], tf.float32)
        HRMS = tf.reshape(HRMS, HRMS_shape)
        A_shape = parsed_features['A_shape']
        A = tf.decode_raw(parsed_features['A'], tf.float32)
        A = tf.reshape(A, A_shape)
        return LRHS, HRMS, A, pos

    pass


if __name__ == '__main__':
    data_num = 0
    net = SpfNet(data_num, sim=True)
    net.stats_graph(tf.get_default_graph())
    net.train_tf()
    # net.test()
    # net.test_piece()
    net.test_piece_tf()
    net.show_final_result()
    # net.show_final_result(recompute=False)
