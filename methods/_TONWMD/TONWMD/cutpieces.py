from functions import checkFile, generateRandomList, reRankfile
import scipy.io as sio
import shutil
import os
from functions import roundNum


def cutCAVEPieces(mat_save_path, piece_save_path, piece_size=32, stride=16,num_end=20):
    '''
    cutting CAVE(first 20 images) into pieces
    :param mat_save_path:
    :param piece_save_path:
    :param piece_size:
    :param stride:
    :param num:
    :return:
    '''
    rows, cols = 512, 512
    num_start = 1
    # num_end = 20
    mat_path = mat_save_path
    piece_save_path = piece_save_path
    count = 0
    checkFile(piece_save_path)
    for i in range(num_start, num_end + 1):
        mat = sio.loadmat(mat_path + '%d.mat' % i)
        X = mat['X']
        Xin = mat['XES']
        for x in range(0, rows - piece_size + stride, stride):
            for y in range(0, cols - piece_size + stride, stride):
                data_piece = Xin[x:x + piece_size, y:y + piece_size, :]
                label_piece = X[x:x + piece_size, y:y + piece_size, :]
                count += 1
                sio.savemat(piece_save_path + '%d.mat' % count, {'data': data_piece, 'label': label_piece})
                print('piece num %d has saved' % count)
        print('%d has finished' % i)
    print('done')
    return count


def generateVerticationSet(train_path, verti_path, num):
    '''
    Randomly select 20% as the verification set
    :param train_path:
    :param verti_path:
    :param num: the number of the patches
    :return:
    '''
    ratio = 0.2
    verti_num = roundNum(num * ratio)
    num_list = []
    checkFile(verti_path)
    generateRandomList(num_list, num, verti_num)
    print(num_list)
    for ind, val in enumerate(num_list):
        # mat = sio.loadmat(train_path+'%d.mat'%val)
        # sio.savemat(verti_path+'%d.mat'%(ind+1),mat)
        try:
            shutil.copy(train_path + '%d.mat' % val, verti_path + '%d.mat' % (ind + 1))
            os.remove(train_path + '%d.mat' % val)
            print('%d has created' % (ind + 1))
        except:
            print('raise error')
    print('veticatication set created')
    print('do rerank train set')
    # rename the left train pieces
    reRankfile(train_path, 'a')
    reRankfile(train_path, '')
    return num-verti_num, verti_num
