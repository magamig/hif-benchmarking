
from clean_util import  para_setting,H_z,HT_y
import os
import torch
from scipy.io import loadmat,savemat
import numpy as np

factor = 8
sz = 512

fft_B,fft_BT = para_setting('gaussian_blur',factor,[sz, sz])
fft_B = torch.cat( (torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)) ,2 ).cuda()
fft_BT = torch.cat( (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)) ,2 ).cuda()

HR_dir =  './CAVE/NSSR_P/test/HSI'
RGB_dir = './CAVE/NSSR_P/test/RGB'
save_dir = './CAVE/NSSR_P/test/LR_8_gaussian/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
imgs = os.listdir(HR_dir)
with torch.no_grad():
    for i in range(len(imgs)):
        print(imgs[i])
        HR = loadmat(os.path.join(HR_dir, imgs[i]))
        HR = torch.FloatTensor(HR['hsi']).permute(2,0,1).unsqueeze(0).cuda()

        RGB = loadmat(os.path.join(RGB_dir, imgs[i]))
        RGB = torch.FloatTensor(RGB['rgb']).permute(2, 0, 1).unsqueeze(0).cuda()

        LR = H_z(HR, factor, fft_B )
        LR = np.asarray(LR.cpu())
        savemat(save_dir + imgs[i], {'lr': LR })

