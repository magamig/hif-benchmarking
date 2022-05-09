import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorly as tl
from skimage.measure import compare_psnr, compare_ssim


def gauss_kernel(row_size, col_size, sigma):
    kernel = cv2.getGaussianKernel(row_size, sigma)
    kernel = kernel * cv2.getGaussianKernel(col_size, sigma).T
    return kernel


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def quality_assessment(out: dict, reference, target, ratio):
    out['cc'] = CC(reference, target)
    out['sam'] = SAM(reference, target)[0]
    out['rmse'] = RMSE(reference, target)
    out['egras'] = ERGAS(reference, target, ratio)
    out['psnr'] = PSNR(reference, target)
    out['ssim'] = SSIM(reference, target)
    return out


def dot(m1, m2):
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def CC(reference, target):
    bands = reference.shape[2]
    out = np.zeros([bands])
    for i in range(bands):
        ref_temp = reference[:, :, i].flatten(order='F')
        target_temp = target[:, :, i].flatten(order='F')
        cc = np.corrcoef(ref_temp, target_temp)
        out[i] = cc[0, 1]
    return np.mean(out)


def SAM(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 1 / (2 ** 52)  # 浮点精度
    prod_scal = dot(reference, target)  # 取各通道相同位置组成的向量进行内积运算
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)  # 二范数乘积矩阵
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps  # 除法避免除数为0
    map = np.arccos(prod_scal / prod_map)  # 求得映射矩阵
    prod_scal = np.reshape(prod_scal, [pixels, 1])
    prod_norm = np.reshape(prod_norm, [pixels, 1])
    z = np.argwhere(prod_norm == 0)[:0]  # 求得prod_norm中为0位置的行号向量
    # 去除这些行，方便后续进行点除运算
    prod_scal = np.delete(prod_scal, z, axis=0)
    prod_norm = np.delete(prod_norm, z, axis=0)
    # 求取平均光谱角度
    angolo = np.sum(np.arccos(prod_scal / prod_norm)) / prod_scal.shape[0]
    # 转换为度数
    angle_sam = np.real(angolo) * 180 / np.pi
    return angle_sam, map


def SSIM(reference, target):
    rows, cols, bands = reference.shape
    mssim = 0
    for i in range(bands):
        mssim += SSIM_BAND(reference[:, :, i], target[:, :, i])
    mssim /= bands
    return mssim


def SSIM_BAND(reference, target):
    return compare_ssim(reference, target, data_range=1.0)


def PSNR(reference, target):
    max_pixel = 1.0
    return 10.0 * np.log10((max_pixel ** 2) / np.mean(np.square(reference - target)))


def RMSE(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sqrt(np.sum((reference - target) ** 2) / pixels)
    return out


def ERGAS(references, target, ratio):
    rows, cols, bands = references.shape
    d = 1 / ratio
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels
    uk = np.mean(tar_temp, axis=0)
    relative_rmse2 = rmse2 / uk ** 2
    total_relative_rmse = np.sum(relative_rmse2)
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)
    return out


class Param(object):
    def __init__(self, data_num):
        self.genPath = 'e:/data_pycode/'
        self.origin_data_path = 'd:/remote sensing data/'
        if data_num == 0:  # cave_data
            self.origin_data_path = self.origin_data_path + 'CAVE/'
            self.mat_save_path = self.genPath + 'cave_data/'
            self.train_start, self.train_end = 1, 20
            self.test_start, self.test_end = 21, 32
            self.total_train_img = self.train_end
            self.total_img = self.test_end
            self.piece_size, self.stride = 64, 16
            self.train_batch_size, self.valid_batch_size = 32, 32
            self.test_stride = self.stride
            self.ratio = 8
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
        # default training info
        self.lr = 1e-3
        self.max_power = 30  # 0.5 * psnr + 0.5 * ssim
        self.epochs = 200

    @staticmethod
    def create_spec_resp(data_num, genPath):
        if data_num == 0:
            file = genPath + 'SRF/D700.mat'  # 377-948
            mat = sio.loadmat(file)
            spec_rng = np.arange(400, 700 + 1, 10)
            spec_resp = mat['spec_resp']
            R = spec_resp[spec_rng - 377, 1:4].T
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

    def read_original_data(self, data_num, input_path, output_path):
        if data_num == 0:
            self.read_cave(input_path, output_path)

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
    def train_valid_piece_save(train_count, test_count, mat_dict, train_data_path, valid_data_path, num_piece, valid_ratio):
        # randomly divide the files into train and validation sets
        if np.random.rand() > valid_ratio:
            folder_num = train_count // num_piece
            check_dir(train_data_path + str(folder_num) + '/')
            sio.savemat(train_data_path + str(folder_num) + '/%d.mat' % (train_count - folder_num * num_piece), mat_dict)
            train_count += 1
        else:
            folder_num = test_count // num_piece
            check_dir(valid_data_path + str(folder_num) + '/')
            sio.savemat(valid_data_path + str(folder_num) + '/%d.mat' % (test_count - folder_num * num_piece), mat_dict)
            test_count += 1
        return train_count, test_count

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
