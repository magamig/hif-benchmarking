from DataReader import readCAVEData, createSimulateData
from functions import get_kernal, getSpectralResponse, checkFile
from cutpieces import cutCAVEPieces, generateVerticationSet
from deepnet.TONWMD import tonwmd
import scipy.io as sio
import numpy as np
from quality_measure import quality_accessment
import time
from optimization import twice_optimization_with_estBR
import tensorflow as tf

# demo for TONWMD on CAVE dataset

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS

# Mode：train, test, creating_data(this is for the first time to train)
tf.app.flags.DEFINE_string('mode', 'creating_data',
                           'train or test or creating_data or cut_pieces or quality_evaluate')
# number of gpus used for training
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
# original data path
tf.app.flags.DEFINE_string('orginal_data_path', 'f:/Fairy/CAVE/',
                           '')
# simulated Data saved path
tf.app.flags.DEFINE_string('mat_save_path', 'temp/CAVEMAT/',
                           '')
# the start number
tf.app.flags.DEFINE_integer('num_start', 1,
                            '')
# the end number
tf.app.flags.DEFINE_integer('num_end', 32,
                            '')
# height of one data HSI
tf.app.flags.DEFINE_integer('height', 512,
                            '')
# width of one data HSI
tf.app.flags.DEFINE_integer('width', 512,
                            '')
# channel of one data HSI
tf.app.flags.DEFINE_integer('channel', 31,
                            '')
# channel of one data MSI
tf.app.flags.DEFINE_integer('ms_channel', 3,
                            '')

# size of Gaussian kernal 8*8
tf.app.flags.DEFINE_integer('kernal_size', 8,
                            '')
# Standard deviation of Gaussian kernal
tf.app.flags.DEFINE_integer('sigma', 8,
                            '')
# each iterlation 64 pieces are trained
tf.app.flags.DEFINE_integer('train_batch_size', 64,
                            '')
# each epoch 16 pieces are verified
tf.app.flags.DEFINE_integer('valid_batch_size', 16,
                            '')
# the size of the pieces is 32*32
tf.app.flags.DEFINE_integer('piece_size', 32,
                            '')
# when cutting pieces,the stride is 16
tf.app.flags.DEFINE_integer('stride', 16,
                            '')
# the downsampling ratio
tf.app.flags.DEFINE_integer('ratio', 8,
                            '')
# in order to select models, we use (psnr + ssim) / 2 to select the model, the initial num is 20
tf.app.flags.DEFINE_float('max_power', 20.0,
                          '')
# test_start for CAVE, the last 12 images are for testing
tf.app.flags.DEFINE_integer('test_start', 21,
                            '')
# test_end
tf.app.flags.DEFINE_integer('test_end', 32,
                            '')
# the count of traing pieces
tf.app.flags.DEFINE_integer('training_num', 15376,
                            '')
# the count of vertication pieces
tf.app.flags.DEFINE_integer('vertication_num', 3844,
                            '')

# training pieces saved path
tf.app.flags.DEFINE_string('train_data_path', 'temp/CAVEMAT/pieces/tonwmd/train/',
                           '')
# vertication pieces saved path
tf.app.flags.DEFINE_string('vertication_data_path', 'temp/CAVEMAT/pieces/tonwmd/valid/',
                           '')
# the path of the models
tf.app.flags.DEFINE_string('model_save_path', 'temp/CAVEMAT/models/tonwmd/',
                           '')
# the path of the CNN predicted results of the test data
tf.app.flags.DEFINE_string('cnn_output_save_path', 'temp/CAVEMAT/cnn_outputs/tonwmd/',
                           '')

# the path of the fusion results of the test data
tf.app.flags.DEFINE_string('output_save_path', 'temp/CAVEMAT/outputs/tonwmd/',
                           '')

# lamb (a regular parameter in pre-optimization is 1e-6 while in post-optimization is 2e-3)
tf.app.flags.DEFINE_float('lamb', 1e-6,
                          '')

# mu (a regular parameter in pre-optimization is 1e-6 while in post-optimization is 2e-3)
tf.app.flags.DEFINE_float('mu', 1e-6,
                          '')

# whether to using the estimated B and R from HySure
tf.app.flags.DEFINE_boolean('isEstimate', False,
                            '')


def train():
    print('begin training -----')
    network = tonwmd(FLAGS.channel, FLAGS.ms_channel, FLAGS.train_data_path, FLAGS.vertication_data_path,
                     FLAGS.model_save_path,
                     FLAGS.mat_save_path, FLAGS.cnn_output_save_path, FLAGS.training_num, FLAGS.vertication_num,
                     train_batch_size=FLAGS.train_batch_size,
                     valid_batch_size=FLAGS.valid_batch_size, piece_size=FLAGS.piece_size, ratio=FLAGS.ratio,
                     maxpower=FLAGS.max_power, test_start=FLAGS.test_start,
                     test_end=FLAGS.test_end, test_height=FLAGS.height,
                     test_width=FLAGS.width)
    network.train()


def test():
    print('predict the results with well_trained CNN-----')
    network = tonwmd(FLAGS.channel, FLAGS.ms_channel, FLAGS.train_data_path, FLAGS.vertication_data_path,
                     FLAGS.model_save_path,
                     FLAGS.mat_save_path, FLAGS.cnn_output_save_path, FLAGS.training_num, FLAGS.vertication_num,
                     train_batch_size=FLAGS.train_batch_size,
                     valid_batch_size=FLAGS.valid_batch_size, piece_size=FLAGS.piece_size, ratio=FLAGS.ratio,
                     maxpower=FLAGS.max_power, test_start=FLAGS.test_start,
                     test_end=FLAGS.test_end, test_height=FLAGS.height,
                     test_width=FLAGS.width)
    network.test()

    FLAGS.lamb = 2e-3
    FLAGS.mu = 2e-3
    if FLAGS.isEstimate:
        # if using the estimated B and R, the algorithm is provided by HySure
        R = sio.loadmat('B_R/R.mat')['R']
        B = sio.loadmat('B_R/B.mat')['B']
        B = np.fft.fft2(B)
    else:
        B = get_kernal(FLAGS.kernal_size, FLAGS.sigma, FLAGS.height, FLAGS.width)
        R = getSpectralResponse()  # come from the nikon camera

    checkFile(FLAGS.output_save_path)
    print('Post-optimization to further improve the performance-------')
    # We use the post-optimization to further improve the performance
    for i in range(FLAGS.test_start, FLAGS.test_end + 1):
        mat = sio.loadmat(FLAGS.mat_save_path + '%d.mat' % i)
        mat2 = sio.loadmat(FLAGS.cnn_output_save_path + '%d.mat' % i)
        Y = mat['Y']
        Z = mat['Z']
        Xcnn = mat2['XCNN']
        # Xes = twice_optimization(Yup, Y, Z, B, R)
        F = twice_optimization_with_estBR(Xcnn, Y, Z, B, R, k=FLAGS.channel, ratio=FLAGS.ratio, lamb=FLAGS.lamb,
                                          mu=FLAGS.mu)
        mat['F'] = F
        sio.savemat(FLAGS.output_save_path + str(i) + '.mat', mat)
        print('F %d has finished' % i)
    print('quality_evaluate-----')
    quality_evaluate()

def creating_data():
    if FLAGS.isEstimate:
        # if using the estimated B and R, the algorithm is provided by HySure
        R = sio.loadmat('B_R/R.mat')['R']
        B = sio.loadmat('B_R/B.mat')['B']
        B = np.fft.fft2(B)
    else:
        B = get_kernal(FLAGS.kernal_size, FLAGS.sigma, FLAGS.height, FLAGS.width)
        R = getSpectralResponse()  # come from the nikon camera

    # to read the original data
    print('read the original HSI-----')
    readCAVEData(FLAGS.orginal_data_path, FLAGS.mat_save_path)
    # create lrhs,hrms and upsampled data
    print('create lrhs,hrms and upsampled data-----')
    createSimulateData(FLAGS.mat_save_path, B, R, ratio=FLAGS.ratio)

    # we use the pre-optimization to obtain the initial hyperspectral images
    print('pre-optimization----------------')
    for i in range(FLAGS.num_start, FLAGS.num_end + 1):
        mat = sio.loadmat(FLAGS.mat_save_path + '%d.mat' % i)
        Y = mat['Y']
        Z = mat['Z']
        Yup = mat['UP']
        # Xes = twice_optimization(Yup, Y, Z, B, R)
        Xin = twice_optimization_with_estBR(Yup, Y, Z, B, R, k=FLAGS.channel, ratio=FLAGS.ratio, lamb=FLAGS.lamb,
                                            mu=FLAGS.mu)
        mat['XES'] = Xin
        sio.savemat(FLAGS.mat_save_path + str(i) + '.mat', mat)
        print('create Xes %d has finished' % i)

def cut_pieces():
    print('cut pieces -----')
    # To train the CNN we need to cut the data into 32*32 pieces, for TONWMD only the XES and X are cut
    count = cutCAVEPieces(FLAGS.mat_save_path, FLAGS.train_data_path, piece_size=FLAGS.piece_size, stride=FLAGS.stride)

    # randomly select 20% of the pieces as the vertication data
    # return the total counts of the vertication pieces and training pieces
    generateVerticationSet(FLAGS.train_data_path, FLAGS.vertication_data_path, count)

def quality_evaluate():
    # evaluate the fusion quality
    out = {}
    average_out = {'cc': 0, 'sam': 0, 'psnr': 0, 'rmse': 0, 'egras': 0, 'ssim': 0}
    for i in range(FLAGS.test_start, FLAGS.test_end + 1):
        mat = sio.loadmat(FLAGS.mat_save_path + '%d.mat' % i)
        reference = mat['X']
        target = sio.loadmat(FLAGS.output_save_path + '%d.mat' % i)['F']
        # target = mat['UP']
        target = np.float32(target)
        quality_accessment(out, reference, target, FLAGS.ratio)
        for key in out.keys():
            average_out[key] += out[key]
        print('image %d has finished' % i)
    for key in average_out.keys():
        average_out[key] /= (FLAGS.test_end - FLAGS.test_start + 1)
    print(average_out)

if __name__ == '__main__':
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'test':  # simple test
            test()
        elif FLAGS.mode == 'creating_data':  # test all
            creating_data()
        elif FLAGS.mode == 'cut_pieces':  # cut pieces
            cut_pieces()
        elif FLAGS.mode == 'train':  # train
            train()
        else:
            quality_evaluate() # evaluate the  fusion quality