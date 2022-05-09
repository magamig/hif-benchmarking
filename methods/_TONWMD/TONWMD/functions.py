import os
import numpy as np
import cv2
import random


def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def reRankfile(path, name):
    '''
    Reorder mat by renaming
    :param path:
    :param name:
    :return:
    '''
    count = 0
    file_list = os.listdir(path)
    for file in file_list:
        try:
            count += 1
            newname = name + str(count)
            print(newname)
            os.rename(path + file, path + '%s.mat' % (newname))
        except:
            print('error')


def getSpectralResponse():
    '''
    spectral response function for CAVE and HARVARD
    :return:
    '''
    R = np.array(
        [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    div = np.sum(R, axis=1)
    div = np.expand_dims(div, axis=-1)
    R = R / div
    return R


def get_kernal(kernal_size, sigma, rows, cols):
    '''
    Generate a Gaussian kernel and make a fast Fourier transform
    :param kernal_size:
    :param sigma:
    :return:
    '''
    # Generate 2D Gaussian filter
    blur = cv2.getGaussianKernel(kernal_size, sigma) * cv2.getGaussianKernel(kernal_size, sigma).T
    psf = np.zeros([rows, cols])
    psf[:kernal_size, :kernal_size] = blur
    # Cyclic shift, so that the Gaussian core is located at the four corners
    B1 = np.roll(np.roll(psf, -kernal_size // 2, axis=0), -kernal_size // 2, axis=1)
    # Fast Fourier Transform
    fft_b = np.fft.fft2(B1)
    # return fft_b
    return fft_b


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.float32(X)


def roundNum(X):
    '''
    rounding
    :param X:
    :return:
    '''
    return int(X + 0.5)


def unfold(x, dim):
    '''
    Matrix shift by dimension
    :param x:
    :param dim:
    :return:
    '''
    new_first_dim = x.shape[dim]
    x = np.swapaxes(x, 0, dim)
    return np.reshape(x, [new_first_dim, -1])


def generateRandomList(numlist: list, maxNum, count):
    '''
    produce needed random list
    :param numlist: random list
    :param maxNum: the max number
    :param count: the count
    :return:
    '''
    i = 0
    while i < count:
        num = random.randint(1, maxNum)
        if num not in numlist:
            numlist.append(num)
            i += 1


def blur_downsample(x, sf, fft_b, height, width):
    '''
    Blur by Fast Fourier Transform
    downsample
    :param x:
    :param sf:
    :param kernal_size:
    :param sigma:
    :return:
    '''
    ch = x.shape[0]
    # rows = height
    # cols = width
    # rows = cols = int(n ** 0.5)
    y = np.zeros([ch, (roundNum(height / sf)) * (roundNum(width / sf))])
    for i in range(ch):
        t = np.reshape(x[i], [height, width], order='F')
        hz = np.real(np.fft.ifft2(np.fft.fft2(t) * fft_b))
        z = hz[::sf, ::sf]
        y[i, :] = z.flatten(order='F')
    return y
