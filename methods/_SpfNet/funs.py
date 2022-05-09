import numpy as np
import tensorflow as tf
import cv2
import os
from skimage.measure import compare_psnr, compare_ssim


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def join_path_list(path: str, names: list, regex: str):
    value = []
    for v in names:
        if os.path.splitext(v)[1] == regex:
            value.append(path + v)
    return value


def gauss_kernel(row_size, col_size, sigma):
    kernel = cv2.getGaussianKernel(row_size, sigma)
    kernel = kernel * cv2.getGaussianKernel(col_size, sigma).T
    return kernel


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def standard(X):
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.float32(X)


def up_sample(X, ratio):
    h, w, c = X.shape
    return cv2.resize(X, (w * ratio, h * ratio), interpolation=cv2.INTER_CUBIC)


def kernel_fft(kernel, rows, cols):
    fft_b = np.zeros([rows, cols])
    fft_b[:kernel.shape[0], :kernel.shape[1]] = kernel
    # Cyclic shift, so that the Gaussian core is located at the four corners
    fft_b = np.roll(np.roll(fft_b, -kernel.shape[0] // 2, axis=0), -kernel.shape[1] // 2, axis=1)
    fft_b = np.fft.fft2(fft_b)
    return fft_b


def intersect(list1, list2):
    list1 = list(list1)
    elem = list(set(list1).intersection(set(list2)))
    elem.sort()
    res = np.zeros(len(elem))
    for i in range(0, len(elem)):
        res[i] = list1.index(elem[i])
    res = res.astype("int32")
    return res


# ======================================================================================================================
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


