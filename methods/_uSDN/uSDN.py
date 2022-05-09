from tensorflow.python.client import device_lib
from utils import *
from pbAuto_uSDN import*
import time
import os
import argparse

parser = argparse.ArgumentParser(description='uSDN for HSI-SR')
parser.add_argument('--cuda', default='0', help='Choose GPU.')
parser.add_argument('--filenum', type=str, default='balloons_ms', help='Image Name.')
parser.add_argument('--load_path', default='_sparse', help='Model Path.')
parser.add_argument('--save_path', default='_sparse')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']= args.cuda
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    # config = tf.ConfigProto(device_count={'GPU':8})
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    loadLRonly = True
    loadLRonly = False
    initp = False
    initp = True
    lr_rate = 0.001
    p_rate = 0.00001
    maxiter = 500000
    tol = 1.5
    vol_r = 0.0001
    sp_r_lsi = 0.001
    sp_r_msi = 0.0001

    num = 12
    ly = 3
    init_num = 15000

    nLRlevel = [ly,ly,ly,ly,ly]
    nHRlevel = [ly,ly,ly,ly,ly]

    file_content = 'data_cave/'+ str(args.filenum) + '.mat'
    load_path = str(args.filenum) + args.load_path + str(num) + '_vol_' + str(vol_r) + '_s'+str(sp_r_lsi)+'/'
    save_path = str(args.filenum) + args.load_path + str(num) + '_vol_' + str(vol_r) + '_s'+str(sp_r_lsi)+'/'

    print('image pair '+str(args.filenum) + ' is processing')

    data = readData(file_content, num)

    # betapan(input data,rate for LR HSI, rate for HR MSI,
    # lr network level, hr network level
    # maximum epoch, is_adam,
    # lr volume rate, lr sparse rate,
    # hr sparse rate,
    # init p or not, tolerence, initnumber)
    auto = betapan(data, lr_rate, p_rate,
                   nLRlevel, nHRlevel,
                   maxiter, True,
                   vol_r,sp_r_lsi,sp_r_msi,initp,config)

    start_time = time.time()
    path = auto.train(load_path, save_path,loadLRonly, tol, init_num)
    # print path
    # auto.generate_hrhsi(save_path, load_path)

    print("--- %s seconds ---" % (time.time() - start_time))
    print('image pair '+str(args.filenum) + ' is done')





if __name__ == "__main__":
    # define main use two __, if use only one_, it will not debug
    main()
