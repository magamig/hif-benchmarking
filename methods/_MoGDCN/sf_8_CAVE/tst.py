
import os
import datetime
import torch
import matplotlib.pyplot as plt
import multiprocessing
import psutil
import numpy as np
from PIL import Image
import sys
import getopt
from network_31 import VSR_CAS
# from dataset_tst import ImageFolder
#from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from util import rgb2ycbcr
import math
import os
import scipy.io as sio
from clean_util import para_setting
from scipy.io import loadmat,savemat
from skimage.measure import compare_ssim
# from util import  PSNR
#########################################################
channel = 31
up_factor = 8
patch_size = 512
delta =3
#########################################################
# --------------------------------------------------------------
data = sio.loadmat('../data/train/NSSR_P')
P    = data['P']
P    = torch.FloatTensor(P)
fft_B,fft_BT = para_setting('gaussian_blur',up_factor,[512, 512],delta=delta)
fft_B = torch.cat( (torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)) ,2 )
fft_BT = torch.cat( (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)) ,2 )
# --------------------------------------------------------------
def convert_to_common_model(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

pretrained = torch.load('./models/eopch_100500_params.pkl')
vsr = torch.nn.DataParallel(VSR_CAS(channel0=channel , factor=up_factor , P=P ,patch_size =patch_size ).cuda())
vsr.load_state_dict(pretrained)
# test data dir
LR_dir = './data/CAVE/test/LR_8_gaussian_'+str(delta)
HR_dir =  './data/CAVE/test/HSI'
RGB_dir = './data/CAVE/test/RGB'

def PSNR(img1, img2):
    mse_sum  = (img1  - img2 ).pow(2)
    mse_loss = mse_sum.mean(2).mean(2) 
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return mse_loss, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def compute_sam(im1, im2):
    im1 = np.reshape(im1,(512*512,31))
    im2 = np.reshape(im2,(512*512,31))
    mole = np.sum(np.multiply(im1, im2), axis=1) 
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    sam = np.rad2deg(np.arccos(((mole+10e-8)/(deno+10e-8)).clip(-1,1)))
    return np.mean(sam)

def compute_ssim(im1,im2): 
    im1 = np.reshape(im1, (512,512,31))
    im2 = np.reshape(im2, (512,512,31))
    n = im1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = compare_ssim(im1[:,:,i], im2[:,:,i])
        ms_ssim += single_ssim
    return ms_ssim/n

def compute_ergas(mse, out):
    out = np.reshape(out, (512*512,31))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (31, 1))
    out_mean = np.reshape(out_mean, (31, 1))
    ergas = 100/8*np.sqrt(np.mean(mse/out_mean**2))                    
    return ergas

imgs = os.listdir(LR_dir)
psnrs      = []
ssims  =[]
sams = []
ergass =[]
pad_LR  = torch.nn.ZeroPad2d(1)
pad_RGB = torch.nn.ZeroPad2d(32)
with torch.no_grad():
    print('=========={}======'.format(len(imgs)))
    for i in range(len(imgs)):
    
        LR = loadmat(os.path.join(LR_dir,imgs[i]))
        LR = torch.FloatTensor(LR['lr']).cuda()
        # LR = pad_LR(LR)
        HR = loadmat(os.path.join(HR_dir, imgs[i]))
        HR = torch.FloatTensor(HR['hsi']).permute(2,0,1).unsqueeze(0).cuda()
        # HR = pad_RGB(HR)
        RGB = loadmat(os.path.join(RGB_dir, imgs[i]))
        RGB = torch.FloatTensor(RGB['rgb']).permute(2, 0, 1).unsqueeze(0).cuda()
        
        res = vsr(LR, RGB)
        res = torch.clamp(res, 0, 1)
        mse, psnr = PSNR(res, HR)
        psnrs.append(psnr)
        res = np.array(res.cpu())
        
        HR    = np.array(HR.cpu())
        mse   = np.array(mse.cpu())
        sam   = compute_sam(res, HR)
        ssim  = compute_ssim(res, HR)
        ergas = compute_ergas(mse, HR)
        sams.append(sam)
        ssims.append(ssim)
        ergass.append(ergas)
        fp = open('./quality.txt', 'a')
        fp.write(imgs[i]  +'\t'+ 'PSNR' +'\t' + 'SAM' +'\t' + 'SSIM' +'\t'+ 'ERGAS'+ '\t' +
                 str(psnr) +'\t' +str(sam)+'\t'+str(ssim)+'\t'+str(ergas)+'\n')
        fp.close()
        print('Image:%s >>>  psnr :%f  , sam : %f  ,ssim : %f , ,ergas : %f'%(imgs[i],psnr,sam,ssim,ergas))

        save_path = './results0615/' + 'delta_' + str(delta) + '/' + imgs[i]
        if not os.path.exists('./results0615/' + 'delta_' + str(delta)):
            os.makedirs('./results0615/' + 'delta_' + str(delta))
        savemat(save_path, {'res': res})



    psnrw_mean = sum(psnrs)/len(psnrs)
    sam_mean   = sum(sams)/len(sams)
    ssim_mean = sum(ssims)/len(ssims)
    ergas_mean = sum(ergass)/len(ergass)
    print('The mean psnr, sam ,ssim, ergas is %f, %f, %f, %f'%(psnrw_mean,sam_mean,ssim_mean,ergas_mean))
    fp=open('./quality.txt','a')
    fp.write('PSNR_mean'  +'\t'+ 'SAM_mean' +'\t' + 'SSIM_mean' +'\t' + 'ERGAS_mean' +
             '\t' + str(psnrw_mean) + '\t' + str(sam_mean) +'\t' + str(ssim_mean)  +'\t'+ str(ergas_mean) + '\n')

    fp.close()










