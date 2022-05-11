from function import *
from SSIM import *
import numpy as np
import scipy.io as sio
import torch
import os
import h5py
import matplotlib.pyplot as plt

CAVE_TestSet = [
    'real_and_fake_apples', 'superballs', 'chart_and_stuffed_toy', 'hairs',  'fake_and_real_lemons',
    'fake_and_real_lemon_slices', 'fake_and_real_sushi', 'egyptian_statue', 'glass_tiles', 'jelly_beans',
    'fake_and_real_peppers', 'clay', 'pompoms', 'watercolors', 'fake_and_real_tomatoes', 'flowers', 'paints',
    'photo_and_face', 'cloth', 'beads'
]

def RMSE_torch(x1, x2):
    MSE = torch.mean((x1-x2)**2)
    return torch.sqrt(MSE)*255


out_path = '../Results/CAVE_K1_S8_40N/'

GT_path = '/Datasets/CAVE/'
names = CAVE_TestSet
#names = os.listdir(GT_path)
total_num = 20

RMSE_SUM = 0
PSNR_SUM = 0
SAM_SUM = 0
SSIM_SUM=0


for name in names:


    data_RE = sio.loadmat(out_path+name+'.mat')

    data_GT = sio.loadmat(GT_path+name+'_ms.mat')
    

    RE = torch.from_numpy(data_RE['RE']).squeeze().type(torch.FloatTensor)
    GT = torch.from_numpy(data_GT['data']).type(torch.FloatTensor)
    GT = GT/(torch.max(GT)-torch.min(GT))
    
    SSIM = ssim_GPU(GT.unsqueeze(0),RE.unsqueeze(0))
    RMSE = RMSE_torch(GT, RE)
    PSNR = PSNR_GPU(GT, RE)
    SAM  = SAM_GPU(GT, RE)

    print('The result of image {0:26} {1:.4f}, {2:.2f}, {3:.2f}, {4:.4f}'.format(name,RMSE,PSNR,SAM,SSIM))
    RMSE_SUM += RMSE
    PSNR_SUM += PSNR
    SAM_SUM  += SAM
    SSIM_SUM += SSIM

print('The average result of this dataset are:{0:.4f}, {1:.2f}, {2:.2f}, {3:.4f}'.format(RMSE_SUM/total_num, PSNR_SUM/total_num, SAM_SUM/total_num, SSIM_SUM/total_num))
