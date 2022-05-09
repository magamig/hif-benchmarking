from tensorflow.python.client import device_lib
from utils_remote import *
from pbAuto_mi  import*
import time
import os
import argparse

# Written by Ying Qu, <yqu3@vols.utk.edu>
# This is a demo code for 'Unsupervised and Unregistered Hyperspectral Image Super-Resolution with Mutual Dirichlet-Net'. 
# The code is for research purposes only. All rights reserved.



# parameter settings
parser = argparse.ArgumentParser(description='Hyperspectral Image Super-Resolution')
parser.add_argument('--cuda', default='0', help='Choose GPU.')
parser.add_argument('--filenum', type=str, default='Pavia_srf_nonrigid1', help='HSI Name.')
parser.add_argument('--load_path', default='_hdly_', help='Model Path.')
parser.add_argument('--save_path', default='_hdly_')
parser.add_argument('--num_hidden', type=int, default=35, help='number of hidden layers')
parser.add_argument('--num_ly', type=int, default=8, help='number of stacked layers')
parser.add_argument('--hsi_lrate', type=float, default=0.001, help='learning rate for hsi')
parser.add_argument('--vol_p', type=float, default=0.001, help='volumn constraint parameter')
parser.add_argument('--mi_p', type=float, default=0.1, help='mutual information parameter')
parser.add_argument('--s_p', type=float, default=0.01, help='sparse parameter')
parser.add_argument('--gen', type=int, default=1, help='generate image')
parser.add_argument('--filename', default='_remote', help='Model Path.')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']= args.cuda
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    # config = tf.ConfigProto(device_count={'GPU':8})
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    
    loadLRonly = True
    loadLRonly = False

    num = args.num_hidden
    ly = args.num_ly
    lhsi_rate = args.hsi_lrate
    # max epoch number
    maxiter = 5000
    tol = 0.0001
    lr_vol_r = args.vol_p
    hsi_mi_r = args.mi_p
    hsi_s_r = args.s_p

    load_path = str(args.filenum)  + args.load_path + str(num) + '_' + 'ly' + str(ly) + \
                '_vol_' + str(lr_vol_r) + '_mi_' + str(hsi_mi_r)+ '_sp_' + str(hsi_s_r)+ args.filename + '/'
    save_dir =  str(args.filenum)  + args.load_path + str(num) + '_' + 'ly' + str(ly) + \
                '_vol_' + str(lr_vol_r) + '_mi_' + str(hsi_mi_r)+ '_sp_' + str(hsi_s_r)+ args.filename + '/'
    filename = 'data/'+ str(args.filenum) + '.mat'

    input = readData(filename, num)
    nLRlevel = [ly,ly+1,ly+2,ly+3,ly+4,ly+5,ly+6]
    nHRlevel = nLRlevel
    # since the image size of the msi is much larger than that of the hsi, we decrease the loss function of the msi to balance the trade-off
    lmsi_rate = 1/input.srfactor

    auto = betapan(input, lhsi_rate, lmsi_rate,
                   nLRlevel, nHRlevel,
                   maxiter, True,
                   lr_vol_r,hsi_mi_r,hsi_s_r, config, save_dir)

    start_time = time.time()
    if args.gen == 0:
        # train the weights in an unsupervised way by reconstructing the hsi and msi
        path = auto.train(load_path, save_dir, loadLRonly, tol)
    else:
        # when the model is saved, the high-resolution hsi can be generated with this function
        auto.testhr(save_dir, load_path)

    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    # define main use two __, if use only one_, it will not debug
    main()
