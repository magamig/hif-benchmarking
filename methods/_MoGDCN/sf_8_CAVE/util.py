import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def PSNR(img1, img2):
    mse = ((img1  - img2 ).pow(2)).mean()#.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# def PSNR(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#        return 100
#     return 20 * math.log10(255.0 / rmse)

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
def rgb2ycbcr_tensor(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    if in_img_type != torch.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = (torch.matmul(img.permute(1,2,0) , torch.tensor([65.481, 128.553, 24.966]))) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt
# path = './test/pad_64/Set14/'
# files = os.listdir(path)
# imgs = []
# for f in files:
#    image = Image.open(path+f).convert('RGB')
#    image = np.array(image)
#    imgs.append(Image.fromarray(rgb2ycbcr(image)))
#     # image = Image.open(path+f).convert('YCbCr')
#     # image = np.array(image)[:,:,0].round().astype(np.uint8)
#     # imgs.append(Image.fromarray(image))
#
# path1 = './results/RGB/Set14/'
# files1 = os.listdir(path1)
# inter_imgs = []
# for f in files1:
#    image = Image.open(path1+f).convert('RGB')
#    image = np.array(image)
#    inter_imgs.append(Image.fromarray(rgb2ycbcr(image)))
#     # image = Image.open(path+f).convert('YCbCr')
#     # image = np.array(image)[:,:,0].round().astype(np.uint8)
#     # imgs.append(Image.fromarray(image))
#
# psnr = []
# for i in range(len(inter_imgs)):
#     x = np.uint8(np.float32(inter_imgs[i]))
#     y = np.uint8(np.float32(imgs[i]))
#     p = PSNR(x.astype(np.float32), y.astype(np.float32))
#     psnr.append(p)
# print(sum(psnr) / len(psnr))
import pypher
import util



def para_setting(kernel_type,sf,sz):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    # elif kernel_type == 'Gaussian':
    #     psf = fspecial('gaussian', 8, 3);
    fft_B = pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT

def H_z(z,fft_B , sf  , sz):
    # x = H_z(z, fft_B, sf, sz)
    n , ch = z.shape
    s0 = sf / 2
    if ch == 1:
        Hz = real(ifft2(torch.fft(reshape(z, sz),2)* fft_B,2));
        x = Hz[1:sf: end, 1: sf:end]
        x = torch.transpose([x[:]])
    else:
        x = np.zeros([ch, int(n / (sf *sf))])
        for i  in range(0,ch):
            Hz = np.real(np.fft.ifft2(np.fft.fft2(np.reshape(z[:,i], sz) )*fft_B) )
            t = Hz[::8 , ::8]
            x[i,:]     =    np.reshape(t , t.shape[0]*t.shape[1])
    return x

def HT_y(y, fft_BT, sf, sz):
    [ch, n] = y.shape
    s0 = sf / 2
    if ch == 1:
        z = zeros(sz);
        z[s0: sf:end, s0: sf:end]  =    reshape(y, floor(sz / sf));
        z = np.real(np.ifft2(np.fft2(z)* fft_BT));
        z = z[:]
    else:
        z = np.zeros([ch, n * sf ** 2])
        t = np.zeros(sz);
        for i  in range(0, ch):

            t[::8, ::8]        =    np.reshape(y[i,:], (np.asarray(sz, dtype=int) / sf).astype(np.int))
            Htz = np.real(np.fft.ifft2(np.fft.fft2(t) * fft_BT))
            z[i,:]                        =    np.reshape(Htz,Htz.shape[0]*Htz.shape[1])
    return z








