import numpy as np
from functions import checkFile
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from Sample import downSample, upSample, spectralDegrade


def readCAVEData(original_data_path, mat_save_path):
    '''
    Read initial CAVE data
    since the original data is standardized we do not repeat it
    :param original_data_path:
    :param mat_save_path:
    :return:
    '''
    path = original_data_path
    hsi = np.zeros([512, 512, 31], dtype=np.float32)
    mat_path = mat_save_path
    checkFile(mat_path)
    count = 0
    for dir in os.listdir(path):
        concrete_path = path + '/' + dir + '/' + dir
        for i in range(31):
            fix = str(i + 1)
            if i + 1 < 10:
                fix = '0' + str(i + 1)
            png_path = concrete_path + '/' + dir + '_' + fix + '.png'
            try:
                hsi[:, :, i] = plt.imread(png_path)
            except:
                img = plt.imread(png_path)
                img = img[:, :, :3]
                img = np.mean(img, axis=2)
                hsi[:, :, i] = img

        count += 1
        print('%d has finished' % count)
        sio.savemat(mat_path + str(count) + '.mat', {'X': hsi})


def createSimulateData(mat_save_path, B, R, ratio=8, num_start = 1, num_end = 32):
    '''
    create simulated data of CAVE
    :param num:
    :return:
    '''
    mat_path = mat_save_path
    for i in range(num_start, num_end + 1):
        mat = sio.loadmat(mat_path + '%d.mat' % i)
        hs = mat['X']
        ms = spectralDegrade(hs, R)
        # lrhs = downSampe2(hs, B, 8)
        lrhs = downSample(hs, B, ratio)
        uplrhs = upSample(lrhs, ratio)
        mat['Z'] = np.float32(ms)
        mat['Y'] = np.float32(lrhs)
        mat['UP'] = np.float32(uplrhs)
        sio.savemat(mat_path + str(i) + '.mat', mat)
        print('%d has finished' % i)
