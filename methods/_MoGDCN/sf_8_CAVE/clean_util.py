import torch
import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pypher
import torch.nn.functional as F
import cv2


def para_setting(kernel_type,sf,sz,delta):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, delta), (cv2.getGaussianKernel(sf, delta)).T)
    fft_B = pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


def H_z(z, factor, fft_B ):
    #     z  [31 , 96 , 96]
    #     ch, h, w = z.shape
    f = torch.rfft(z, 2, onesided=False)
    # -------------------complex myltiply-----------------#
    if len(z.shape)==3:
        ch , h, w = z.shape
        fft_B = fft_B.unsqueeze(0).repeat(ch,1,1,1)
        M = torch.cat(( (f[:,:,:,0]*fft_B[:,:,:,0]-f[:,:,:,1]*fft_B[:,:,:,1]).unsqueeze(3) ,
                        (f[:,:,:,0]*fft_B[:,:,:,1]+f[:,:,:,1]*fft_B[:,:,:,0]).unsqueeze(3) )  ,3)
        Hz = torch.irfft(M, 2, onesided=False)
        x = Hz[:, int(factor//2)::factor ,int(factor//2)::factor]
    elif len(z.shape)==4:
        bs,ch,h,w = z.shape
        fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs,ch, 1, 1, 1)
        M = torch.cat((  (f[:,:,:,:,0]*fft_B[:,:,:,:,0]-f[:,:,:,:,1]*fft_B[:,:,:,:,1]).unsqueeze(4) ,
                         (f[:,:,:,:,0]*fft_B[:,:,:,:,1]+f[:,:,:,:,1]*fft_B[:,:,:,:,0]).unsqueeze(4) ), 4)
        Hz = torch.irfft(M, 2, onesided=False)
        x = Hz[: ,: , int(factor//2)::factor ,int(factor//2)::factor]
    return x
    # s0 = sf / 2
    # if ch == 1:
    #     Hz = real(ifft2(torch.fft(reshape(z, sz),2)* fft_B,2))
    #     x = Hz[1:sf: end, 1: sf:end]
    #     x = torch.transpose([x[:]])
    # else:


        # torch.rfft(z,2,onesided=False)
        # x = np.zeros([ch, int(n / (sf *sf))])
        # x = np.zeros([ch, int( h/sf), int( w/sf)])
        #
        # for i  in range(0,ch) :
        #     Hz = np.real(np.fft.ifft2(np.fft.fft2(np.reshape(z[i,:,:], sz) )*fft_B) )
        #     t = Hz[::factor , ::factor]
        #     x[i,:,:]     =    t
    # return x

def HT_y(y, sf, fft_BT):
    if len(y.shape) == 3:
        ch, w, h = y.shape
        # z = torch.zeros([ch, w*sf ,h*sf])
        # z[:,::sf, ::sf] = y
        z = F.pad(y, [0, 0, 0, 0, 0, sf * sf - 1], "constant", value=0)
        z = F.pixel_shuffle(z, upscale_factor=sf).view(bs, ch, w * sf, h * sf)

        f = torch.rfft(z , 2 ,onesided = False)
        fft_BT = fft_BT.unsqueeze(0).repeat(ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                       (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
        Hz = torch.irfft(M, 2, onesided=False)
    elif len(y.shape) == 4:
        bs ,ch ,w ,h = y.shape
        # z = torch.zeros([bs ,ch ,sf*w ,sf*w])
        # z[:,:,::sf,::sf] = y
        z = y.view(-1, 1, w, h)
        z = F.pad(z, [0, 0, 0, 0, 0, sf*sf-1, 0, 0], "constant", value=0)
        z = F.pixel_shuffle(z, upscale_factor=sf).view(bs ,ch ,w*sf ,h*sf)

        f = torch.rfft(z, 2, onesided=False)
        fft_BT = fft_BT.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, :, 0] * fft_BT[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_BT[:, :, :, :, 1]).unsqueeze(4),
                       (f[:, :, :, :, 0] * fft_BT[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_BT[:, :, :, :, 0]).unsqueeze(4)), 4)
        Hz = torch.irfft(M, 2, onesided=False)
    return Hz
    # [ch, n] = y.shape
    # s0 = sf / 2
    # if ch == 1:
    #     z = zeros(sz);
    #     z[s0: sf:end, s0: sf:end]  =    reshape(y, floor(sz / sf));
    #     z = np.real(np.ifft2(np.fft2(z)* fft_BT));
    #     z = z[:]
    # else:
    #     z = np.zeros([ch, n * sf ** 2])
    #     t = np.zeros(sz);
    #     for i  in range(0, ch):
    #
    #         t[::8, ::8]        =    np.reshape(y[i,:], (np.asarray(sz, dtype=int) / sf).astype(np.int))
    #         Htz = np.real(np.fft.ifft2(np.fft.fft2(t) * fft_BT))
    #         z[i,:]                        =    np.reshape(Htz,Htz.shape[0]*Htz.shape[1])
    # return z

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s

def delta_time(datetime1, datetime2):
    if datetime1 > datetime2:
        datetime1, datetime2 = datetime2, datetime1
    second = 0
    second += (datetime2.day - datetime1.day) * 24 * 3600
    second += (datetime2.hour - datetime1.hour) * 3600
    second += (datetime2.minute - datetime1.minute) * 60
    second += (datetime2.second - datetime1.second)
    return second

def save_checkpoint(net, optimizer, epoch, losses, savepath):
    save_json = {
        # 'cuda_flag': net.cuda_flag,
        # 'h': net.height,
        # 'w': net.width,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)

def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # net.cuda_flag = checkpoint['cuda_flag']
    # net.height = checkpoint['h']
    # net.width = checkpoint['w']
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    return net, optimizer, start_epoch, losses

def grad_img(img):

    h = img[..., :-1, 1:]-img[..., :-1, :-1]
    w = img[..., 1:, :-1]-img[..., :-1, :-1]

    return torch.cat( (h, w), dim= -3)
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
       return 100
    return 20 * math.log10(255.0 / rmse)

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def creat_P():


    P = [[2. ,1. , 1.  ,1.  ,1 , 1.  ,0. , 0.  ,0. , 0.  ,0. , 0. , 0. , 0. , 0.  ,0.  ,2. , 6. ,11. ,17. ,21. ,22. ,21. ,20. ,20. ,19. ,19., 18. ,18. ,17., 17.],
         [1. ,1. , 1.  ,1.  ,1. , 1.  ,2. , 4.,  6.,  8.,  11., 16., 19., 21., 20., 18., 16., 14.,11.,  7., 5.,  3.,  2.,  2.,  1.,  1.,   2.,  2.,  2.,  2., 2.],
         [7. ,10., 15. ,19. ,25. ,29. ,30. ,29. ,27., 22. ,16. , 9. ,2.  ,0.  , 0. , 0. , 0. , 0. ,0. , 0. ,1. , 1. ,  1. , 1.,  1. , 1. , 1.  ,1. , 1.,  1.  ,1.]]
    P = np.array(P)

    for band in range(3):
        div = sum(P[band,:])
        for i in range(31):
            P[band, i] = P[band, i] / div
    return torch.FloatTensor(P)


