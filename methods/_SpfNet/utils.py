# This file contains two classes for deep learning fusion
import numpy as np
import tensorflow as tf
import tensorly as tl
import scipy.io as sio
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import os
import math
import cv2
import platform

from SpfNet.funs import _int64_features, _bytes_features
from SpfNet.funs import gauss_kernel, check_dir, standard, join_path_list, intersect
from SpfNet.funs import quality_assessment


class Param(object):
    def __init__(self, data_num):
        sysstr = platform.system()
        if sysstr == 'Windows':
            self.genPath = 'D:/PyCode/PySCI/SpfNet/'
            # d:/remote sensing data/CAVE/balloons_ms/balloons_ms/balloons_ms_01.png
            # d:/remote sensing data/harvard/img1.mat
            self.origin_data_path = 'd:/remote sensing data/'
        if sysstr == 'Linux':
            self.genPath = '/home/amax/PycharmProjects/???/'
            self.origin_data_path = '/home/amax/PycharmProjects/???/'
        if data_num == 0:  # cave data
            self.origin_data_path = self.origin_data_path + 'CAVE/'
            self.mat_save_path = self.genPath + 'cave_data/'
            self.train_start, self.train_end = 1, 20
            self.test_start, self.test_end = 21, 32
            self.total_train_img = self.train_end
            self.total_img = self.test_end
            self.piece_size, self.stride = 64, 16
            self.train_batch_size, self.valid_batch_size = 32, 32
            self.block = self.total_train_img
            self.test_stride = self.stride
            self.ratio = 8
        if data_num == 1:  # harvard data
            self.origin_data_path = self.origin_data_path + 'harvard/'
            self.mat_save_path = self.genPath + 'harvard_data/'
            self.train_start, self.train_end = 1, 30
            self.test_start, self.test_end = 31, 50
            self.total_train_img = self.train_end
            self.total_img = self.test_end
            self.piece_size, self.stride = 64, 48
            self.train_batch_size, self.valid_batch_size = 32, 32
            self.block = self.total_train_img
            self.test_stride = 16
            self.ratio = 8
        if data_num == 2:  # center data
            self.origin_data_path = self.origin_data_path + 'center/'
            self.mat_save_path = self.genPath + 'center_data/'
            self.train_start, self.train_end = 1, 2
            self.test_start, self.test_end = 3, 3
            self.total_train_img = self.train_end
            self.total_img = self.test_end
            self.piece_size, self.stride = 64, 8
            self.train_batch_size, self.valid_batch_size = 8, 8
            self.block = self.total_train_img
            self.test_stride = 4
            self.ratio = 8
        if data_num == 11:  # WV2 data
            self.origin_data_path = self.origin_data_path + 'WV2/'
            self.mat_save_path = self.genPath + 'wv2_data/'
            self.train_start, self.train_end = 1, 2
            self.test_start, self.test_end = 3, 4
            self.total_train_img = self.train_end
            self.total_img = self.test_end
            self.piece_size, self.stride = 32, 4
            self.train_batch_size, self.valid_batch_size = 32, 32
            self.test_stride = self.stride
            self.block = 10
            self.ratio = 4
        name = self.__class__.__name__
        print('%s is running' % name)
        self.sim_save_path = self.mat_save_path + "sim/"
        self.param_path = self.mat_save_path + name + '-net/'
        self.est_save_path = self.mat_save_path + name + "-net/est/"
        self.train_data_path = self.mat_save_path + name + "-net/train/"
        self.valid_data_path = self.mat_save_path + name + "-net/valid/"
        self.model_save_path = self.mat_save_path + name + "-net/model/"
        self.output_save_path = self.mat_save_path + name + "-net/output/"
        self.test_data_path = self.param_path + 'test/'
        self.sigma = (1 / (2 * 2.7725887 / self.ratio ** 2)) ** 0.5
        self.kerSize = 2 * self.ratio - 1
        self.hs_snr, self.ms_snr = 30, 40

    pass


class FusionNet(Param):
    def __init__(self, data_num, sim=True):
        super().__init__(data_num)
        if sim is True:
            self.B = gauss_kernel(self.kerSize, self.kerSize, sigma=self.sigma)
            self.R = self.create_spec_resp(data_num, self.genPath)
            if not os.path.exists(self.sim_save_path + 'info.mat'):
                check_dir(self.sim_save_path)
                sio.savemat(self.sim_save_path + 'info.mat', {'B': self.B, 'R': self.R})
            self.read_original_data(data_num, self.origin_data_path, self.mat_save_path)  # HS
            self.hs_bands, self.ms_bands = self.create_simulate_data(self.mat_save_path, self.sim_save_path,
                                                                     self.B, self.R, self.ratio,
                                                                     self.hs_snr, self.ms_snr)  # HS/LRHS/HRMS
        else:
            self.hs_bands, self.ms_bands = self.read_sim_data(data_num, self.origin_data_path, self.sim_save_path)
        # default training info
        self.lr = 1e-3
        self.l2_decay = 1e-4
        self.max_lr, self.min_lr, self.decay_speed = 1e-3, 1e-5, 45
        self.max_power = 15  # 0.5 * psnr + 0.5 * ssim
        self.current_epoch, self.total_epoch = 0, 100  # total_epoch should not too large for the memory usage
        self.current_restart, self.restart_epoch = 0, 1  # one can enlarge restart_epoch for large total_epoch

    # ------------------------------------------- train and test -------------------------------------------------

    def train_data_tf(self, repeat=None, buffer_size=None):
        if repeat is None:
            repeat_num = self.total_epoch
        else:
            repeat_num = repeat
        if buffer_size is None:
            buffer_size = self.train_batch_size
        file_list = os.listdir(self.train_data_path)
        file_list = join_path_list(self.train_data_path, file_list, regex='.tfrecords')
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(self._parse_record, num_parallel_calls=None)
        dataset = dataset.shuffle(buffer_size=buffer_size).repeat(repeat_num)
        dataset = dataset.batch(batch_size=self.train_batch_size).prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def valid_data_tf(self):
        file_list = os.listdir(self.valid_data_path)
        file_list = join_path_list(self.valid_data_path, file_list, regex='.tfrecords')
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(self._parse_record, num_parallel_calls=None)
        dataset = dataset.batch(batch_size=self.valid_batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    @staticmethod
    def _parse_record(example):  # return HS/LRHS/HRMS, one can override it
        features = {'HS_shape': tf.FixedLenFeature([3], tf.int64),
                    'LRHS_shape': tf.FixedLenFeature([3], tf.int64),
                    'HRMS_shape': tf.FixedLenFeature([3], tf.int64),
                    'HS': tf.FixedLenFeature([], tf.string),
                    'LRHS': tf.FixedLenFeature([], tf.string),
                    'HRMS': tf.FixedLenFeature([], tf.string)}
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
        return HS, LRHS, HRMS

    # ------------------------------------------------ learning rate -------------------------------------------
    def decrease_lr(self):
        # exp(-5) = 0.0067
        self.lr = self.min_lr + (self.max_lr - self.min_lr) * math.exp(- self.current_epoch / self.decay_speed)

    def decrease_lr_v0(self):
        self.lr = 1e-4

    def decrease_lr_v2(self):
        if 0 <= self.current_epoch < 20:
            self.lr = 1e-3
        elif 20 <= self.current_epoch < 50:
            self.lr = 5 * 1e-4
        elif 50 <= self.current_epoch < 100:
            self.lr = 1e-4
        elif 100 <= self.current_epoch < 150:
            self.lr = 5 * 1e-5
        else:
            self.lr = 1e-5

    # --------------------------------------------- show results -----------------------------------------------
    def show_final_result(self, recompute=True):
        out = {}
        average_out = {'rmse': 0, 'psnr': 0, 'sam': 0, 'ssim': 0, 'cc': 0, 'egras': 0, 'TM': 0}
        for i in range(self.test_start, self.test_end + 1):
            mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
            reference = mat['HS']
            mat = sio.loadmat(self.output_save_path + '%d.mat' % i)
            target = mat['F']
            target = np.float32(target)
            if recompute is True:
                quality_assessment(out, reference, target, self.ratio)
                mat.update(out)
                sio.savemat(self.output_save_path + '%d.mat' % i, mat)
            # else:
            #     out = mat
            for key in average_out.keys():
                average_out[key] += mat[key]
            print('image %d has finished' % i)
        for key in average_out.keys():
            average_out[key] /= (self.test_end - self.test_start + 1)
        print(average_out)
        return average_out

    def show_result(self, test_path, test_name):
        out = {}
        average_out = {'rmse': 0, 'psnr': 0, 'sam': 0, 'ssim': 0, 'cc': 0, 'egras': 0}
        num_start, num_end = self.test_start, self.test_end + 1
        # num_start, num_end = 1, 2+ 1
        for i in range(num_start, num_end):
            mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
            reference = mat['HS']
            target = sio.loadmat(test_path + '%d.mat' % i)[test_name]
            target = np.float32(target)
            quality_assessment(out, reference, target, self.ratio)
            for key in out.keys():
                average_out[key] += out[key]
            print('image %d has finished' % i)
        for key in average_out.keys():
            average_out[key] /= (num_end - num_start)
        print(average_out)

    # -------------------------------------------- original data ----------------------------------------------
    def read_original_data(self, data_num, input_path, output_path):
        if data_num == 0:
            self.read_cave(input_path, output_path)
        if data_num == 1:
            self.read_harvard(input_path, output_path)
        if data_num == 2:
            self.read_center(input_path, output_path)

    def read_sim_data(self, data_num, input_path, sim_path):
        hs_bands, ms_bands = 0, 0
        if data_num == 11:
            hs_bands, ms_bands = self.read_wv2(input_path, sim_path)
        return hs_bands, ms_bands

    @staticmethod
    def read_wv2(input_path, sim_path):  # please give the names of files
        if os.path.exists(sim_path + '1.mat'):
            print('WV2: xx exists!')
            mat = sio.loadmat(sim_path + '1.mat')
            hs_bands = mat['LRHS'].shape[-1]
            ms_bands = mat['HRMS'].shape[-1]
            return hs_bands, ms_bands
        check_dir(sim_path)
        dir_name = 'WV2_train_d1.mat'
        mat = sio.loadmat(input_path + dir_name)
        sio.savemat(sim_path + '1.mat', {'HS': mat['I_REF'], 'LRHS': mat['I_HSI'], 'HRMS': mat['I_MSI']})
        print('Simulate: 1.mat has finished')
        dir_name = 'WV2_train_d2.mat'
        mat = sio.loadmat(input_path + dir_name)
        sio.savemat(sim_path + '2.mat', {'HS': mat['I_REF'], 'LRHS': mat['I_HSI'], 'HRMS': mat['I_MSI']})
        print('Simulate: 2.mat has finished')
        dir_name = 'WV2_test_d.mat'
        mat = sio.loadmat(input_path + dir_name)
        sio.savemat(sim_path + '3.mat', {'HS': mat['I_REF'], 'LRHS': mat['I_HSI'], 'HRMS': mat['I_MSI']})
        print('Simulate: 3.mat has finished')
        dir_name = 'WV2_test.mat'
        mat = sio.loadmat(input_path + dir_name)
        LRHS = mat['I_HSI']
        HRMS = mat['I_MSI']
        HS = np.ones([HRMS.shape[0], HRMS.shape[1], LRHS.shape[2]])
        sio.savemat(sim_path + '4.mat', {'HS': HS, 'LRHS': LRHS, 'HRMS': HRMS})
        print('Simulate: 4.mat has finished')
        return LRHS.shape[2], HRMS.shape[2]

    @staticmethod
    def read_harvard(input_path, mat_path):
        if os.path.exists(mat_path + '50.mat'):
            print('Harvard: xx exists!')
            return
        check_dir(mat_path)
        count = 0
        for dir_name in os.listdir(input_path):
            count += 1
            hs = sio.loadmat(input_path + dir_name)['ref']
            hs = standard(hs)
            sio.savemat(mat_path + "%d.mat" % count, {'HS': hs})
            print('Harvard: %d has finished' % count)

    @staticmethod
    def read_cave(input_path, output_path):
        if os.path.exists(output_path + '32.mat'):
            print('CAVE: xx exists!')
            return
        check_dir(output_path)
        rows, cols, bands = 512, 512, 31
        hsi = np.zeros([rows, cols, bands], dtype=np.float32)
        count = 0
        for dir_name in os.listdir(input_path):
            concrete_path = input_path + dir_name + '/' + dir_name
            for i in range(bands):
                fix = str(i + 1)
                if i + 1 < 10:
                    fix = '0' + str(i + 1)
                png_path = concrete_path + '/' + dir_name + '_' + fix + '.png'
                try:
                    hsi[:, :, i] = plt.imread(png_path)
                except ValueError:
                    img = plt.imread(png_path)
                    img = img[:, :, :3]
                    img = np.mean(img, axis=2)
                    hsi[:, :, i] = img
            count += 1
            print('CAVE: %d has finished' % count)
            sio.savemat(output_path + str(count) + '.mat', {'HS': hsi})

    @staticmethod
    def read_center(input_path, output_path):
        if os.path.exists(output_path + '3.mat'):
            print('Center: xx exists!')
            return
        check_dir(output_path)
        count = 0
        for dir_name in os.listdir(input_path):
            count += 1
            hs = sio.loadmat(input_path + dir_name)['ref']
            hs = standard(hs)
            sio.savemat(output_path + "%d.mat" % count, {'HS': hs})
            print('Center: %d has finished' % count)

    @staticmethod
    def create_spec_resp(data_num, genPath):
        if data_num == 0:
            file = genPath + 'srf/D700.mat'  # 377-948
            mat = sio.loadmat(file)
            spec_rng = np.arange(400, 700 + 1, 10)
            spec_resp = mat['spec_resp']
            R = spec_resp[spec_rng - 377, 1:4].T
        if data_num == 1:
            file = genPath + 'srf/D700.mat'  # 377-948
            mat = sio.loadmat(file)
            spec_rng = np.arange(420, 720 + 1, 10)
            spec_resp = mat['spec_resp']
            R = spec_resp[spec_rng - 377, 1:4].T
        if data_num == 2:
            band = 102
            file = genPath + 'srf/ikonos.mat'  # 350 : 5 : 1035
            mat = sio.loadmat(file)
            spec_rng = np.arange(430, 861)
            spec_resp = mat['spec_resp']
            ms_bands = range(1, 5)
            valid_ik_bands = intersect(spec_resp[:, 0], spec_rng)
            no_wa = len(valid_ik_bands)
            # Spline interpolation
            xx = np.linspace(1, no_wa, band)
            x = range(1, no_wa + 1)
            R = np.zeros([5, band])
            for i in range(0, 5):
                ipo3 = spi.splrep(x, spec_resp[valid_ik_bands, i + 1], k=3)
                R[i, :] = spi.splev(xx, ipo3)
            R = R[ms_bands, :]
        c = 1 / np.sum(R, axis=1)
        R = np.multiply(R, c.reshape([c.size, 1]))
        return R

    @staticmethod
    def create_simulate_data(input_path, output_path, B, R, ratio, hs_snr, ms_snr, noise=True):
        if os.path.exists(output_path + '1.mat'):
            print('Simulate: xx exists!')
            mat = sio.loadmat(output_path + '1.mat')
            hs_bands = mat['LRHS'].shape[-1]
            ms_bands = mat['HRMS'].shape[-1]
            return hs_bands, ms_bands
        check_dir(output_path)
        for dir_name in os.listdir(input_path):
            if os.path.isfile(input_path + dir_name):
                mat = sio.loadmat(input_path + dir_name)
                hs = mat['HS']
                # control size of image according to ratio
                hs = hs[0: hs.shape[0] // ratio * ratio, 0: hs.shape[1] // ratio * ratio, :]
                mat['HS'] = hs
                ms = tl.tenalg.mode_dot(hs, R, mode=2)
                # add noise for ms
                ms_sig = (np.sum(np.power(ms.flatten(), 2)) / (10 ** (ms_snr / 10)) / ms.size) ** 0.5
                np.random.seed(1)
                if noise is True:
                    ms = np.add(ms, ms_sig * np.random.randn(ms.shape[0], ms.shape[1], ms.shape[2]))
                # blur
                lrhs = cv2.filter2D(hs, -1, B, borderType=cv2.BORDER_REFLECT)
                # add noise for hs
                hs_sig = (np.sum(np.power(lrhs.flatten(), 2)) / (10 ** (hs_snr / 10)) / lrhs.size) ** 0.5
                np.random.seed(0)
                if noise is True:
                    lrhs = np.add(lrhs, hs_sig * np.random.randn(lrhs.shape[0], lrhs.shape[1], lrhs.shape[2]))
                # down sampling
                lrhs = lrhs[0::ratio, 0::ratio, :]
                mat['HRMS'] = np.float32(ms)
                mat['LRHS'] = np.float32(lrhs)
                sio.savemat(output_path + dir_name, mat)
                print('Simulate: %s has finished' % dir_name)
        mat = sio.loadmat(output_path + '1.mat')
        hs_bands = mat['LRHS'].shape[-1]
        ms_bands = mat['HRMS'].shape[-1]
        return hs_bands, ms_bands

    # ------------------------------ cut piece --------------------------------------

    @staticmethod
    def train_valid_piece_save(train_count, test_count, mat_dict, train_data_path, valid_data_path, block, valid_ratio):
        # randomly divide the files into train and validation sets
        if np.random.rand() > valid_ratio:
            train_count += 1
            block_name = str(np.random.random_integers(1, block))
            check_dir(train_data_path + block_name + '/')
            sio.savemat(train_data_path + block_name + '/%d.mat' % train_count, mat_dict)
        else:
            test_count += 1
            block_name = str(np.random.random_integers(1, np.ceil(block * valid_ratio)))
            check_dir(valid_data_path + block_name + '/')
            sio.savemat(valid_data_path + block_name + '/%d.mat' % test_count, mat_dict)
        return train_count, test_count

    def train_valid_piece_tf(self, train_data_path, valid_data_path):
        folders = os.listdir(train_data_path)
        for dir_name in folders:
            input_path = train_data_path + dir_name + '/'
            output_file = train_data_path + dir_name + '.tfrecords'
            if os.path.isdir(input_path):
                self.files_to_tf(input_path, output_file)
                # shutil.rmtree(input_path, ignore_errors=True)
                print(input_path, output_file)
        folders = os.listdir(valid_data_path)
        for dir_name in folders:
            input_path = valid_data_path + dir_name + '/'
            output_file = valid_data_path + dir_name + '.tfrecords'
            if os.path.isdir(input_path):
                self.files_to_tf(valid_data_path + dir_name + '/', output_file)
                # shutil.rmtree(input_path, ignore_errors=True)
                print(input_path, output_file)

    def files_to_tf(self, input_path, output_file):
        writer = tf.python_io.TFRecordWriter(output_file)
        for dir_name in os.listdir(input_path):
            mat = sio.loadmat(input_path + dir_name)
            feature = self.get_feature_mat(mat)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def get_feature_mat(mat):
        feature = {}
        for key in mat.keys():
            data = mat[key]
            if type(data) is np.ndarray:  # other types?
                data = data.astype(np.float32)
                data_shape = data.shape
                data = data.tostring()
                feature[key] = _bytes_features([data])
                feature[key + '_shape'] = _int64_features(list(data_shape))
        return feature

    # --------------------------------------------------------------------
    # one must add weights by tf.add_to_collection('loss', weight)
    def build_loss(self, label, output, l2_decay=None, norm='l2', sad=False, lam=1e-2):
        loss = 0
        if norm == 'l2':
            # L2-norm as the spatial loss
            loss += tf.reduce_mean(tf.square(label - output), name='mse')
        if norm == 'l1':
            # L1-norm as the spatial loss
            loss += tf.reduce_mean(tf.abs(label - output), name='mse')
        if l2_decay is 'loss':
            # weight_decay to suppress the weights
            weight_list = tf.get_collection('loss')
            l2_norm_losses = [tf.nn.l2_loss(w) for w in weight_list]
            l2_norm_loss = self.l2_decay * tf.add_n(l2_norm_losses)
            loss += l2_norm_loss
        if l2_decay is 'layer':
            l2_norm_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_norm_loss = tf.add_n(l2_norm_losses)
            loss += l2_norm_loss
        if sad is True:
            loss += lam * self.tf_sam(label, output)
        return loss

    @staticmethod
    def tf_sam(label, output):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(label), axis=-1))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(output), axis=-1))
        xy_norm = tf.multiply(x_norm, y_norm)
        xy = tf.reduce_sum(tf.multiply(label, output), axis=-1)
        dist = tf.reduce_mean(tf.acos(tf.div(xy, xy_norm + 1e-8)))
        dist = tf.scalar_mul(180.0 / np.pi, dist)
        return dist

    @staticmethod
    def build_psnr_ssim(label, output):
        psnr_sum = tf.reduce_sum(tf.image.psnr(label, output, max_val=1.0))
        ssim_sum = tf.reduce_sum(tf.image.ssim(label, output, max_val=1.0))
        return psnr_sum, ssim_sum

    @staticmethod
    def init_device():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    # ----------------------------------------- toolkit --------------------------------------------------
    @staticmethod
    def stats_graph(graph, verb=True):
        # graph = tf.get_default_graph()
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        if verb is True:
            print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
        return flops.total_float_ops, params.total_parameters


