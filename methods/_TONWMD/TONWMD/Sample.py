from functions import unfold, blur_downsample
import numpy as np
import cv2


def downSample(X, B, ratio):
    '''
    downsample using fft
    :param X:
    :param B:
    :param ratio:
    :return:
    '''
    r, c, _ = X.shape
    X = unfold(X, 2)
    Y = blur_downsample(X, ratio, B, r, c)
    Y = np.reshape(Y.T, [r // ratio, c // ratio, -1], order='F')
    return Y


def upSample(X, ratio=8):
    '''
    upsample using cubic
    :param X:
    :param ratio:
    :return:
    '''
    h, w, c = X.shape
    return cv2.resize(X, (w * ratio, h * ratio), interpolation=cv2.INTER_CUBIC)


def spectralDegrade(X, R):
    '''
    spectral downsample
    :param X:
    :param R:
    :return:
    '''
    height, width, bands = X.shape
    X = np.reshape(X, [-1, bands], order='F')
    Z = np.dot(X, R.T)
    Z = np.reshape(Z, [height, width, -1], order='F')
    return Z
