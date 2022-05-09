from PSNR import PSNR
import torch
from SpaNet import SpaNet
from SpeNet import SpeNet
import argparse
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import itertools
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
import visdom
from torch.utils import data
from Hyper_loader_2 import Hyper_dataset
from torchvision.utils import make_grid
# from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import skimage.measure as skm
import numpy as np
from SSIM import SSIM
# from apex import amp
import os
import cv2
import skimage
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())
def loss_spa(hsi,hsi_t,Mse):
    loss = Mse(hsi[1],hsi_t[1]) + Mse(hsi[2],hsi_t[2])
    return loss
state_dict = torch.load('/home/zhu_19/Hyperspectral_image/resume/model/state_dicr_200.pkl')
def get_spe_gt(hsi):
    output=[]
    output.append(hsi)
    index = [np.array(list(range(8)))*4,np.array(list(range(16)))*2]
    index[0][-1] = 30
    index[1][-1] = 30
    output.append(hsi[:,index[0],:,:])
    output.append(hsi[:,index[1],:,:])
    return output
# def load_psnr(path):
#     state_dict = torch.load(path)

def load_model(model,mode_dict):
    # state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in mode_dict.items():
        name = k[7:] # remove `module.`
        # name = k # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
if __name__ == '__main__':
    spanet = SpaNet().cuda()
    spenet = SpeNet().cuda()
    spanet = load_model(spanet,state_dict['spanet'])
    spenet = load_model(spenet,state_dict['spenet'])
    # spanet.load_state_dict(state_dict['spanet'])
    # spenet.load_state_dict(state_dict['spenet'])
    spanet.eval()
    spenet.eval()
    MaeLoss = torch.nn.L1Loss()
    Hyper_test = Hyper_dataset(output_shape=128,Training_mode='Test',data_name='CAVE')
    Hyper_test = DataLoader(Hyper_test,batch_size=1,shuffle=True,num_workers=4)

    with torch.no_grad():
        loss_t = []
        psnr_g = []
        ssim_log = []
        # RMSE = []
        i = 0
        for hsi,hsi_g,hsi_resize,msi in Hyper_test:
            spanet.eval()
            spenet.eval()
            hsi_g = hsi_g.cuda().float()
            hsi_resize = hsi_resize.cuda().float()
            hsi_resize = torch.nn.functional.interpolate(hsi_resize,scale_factor=(8,8),mode='bilinear')
            hsi = [a.cuda().float() for a in hsi]
            msi = [a.cuda().float() for a in msi]
            hsi_spa = spanet(hsi[0],msi)
            hsi_spe = spenet(hsi_resize,msi[-1])
            loss_a = loss_spa(hsi_spa,hsi,Mse = MaeLoss)
            hsi_e = get_spe_gt(hsi_g)
            loss_e = loss_spa(hsi_spe,hsi_e,Mse = MaeLoss)
            hsi_1 = hsi_spa[-1][:,:31,:,:]
            hsi_2 = hsi_spe[-1][:,:31,:,:]
            weight_1 = hsi_spa[-1][:,31:,:,:]
            weight_2 = hsi_spe[-1][:,31:,:,:]
            weight = torch.cat((weight_1[:,:,:,:,None],weight_2[:,:,:,:,None]),-1)
            weight = F.softmax(weight,dim=-1)
            weight_1 = weight[:,:,:,:,0]
            weight_2 = weight[:,:,:,:,1]
            fout = hsi_1 * weight_1 + hsi_2 * weight_2
            loss = MaeLoss(fout,hsi_g)+ loss_a + loss_e+MaeLoss(hsi_1,hsi_g)+MaeLoss(hsi_2,hsi_g)
            loss_t.append(loss.item())
            # ssim = SSIM(fout,hsi_g).mean(1)[0,0,0].item()
            hsi_g_ = (hsi_g*(2**16-1)).int().detach().cpu().numpy()[0,:,:,:]
            fout_ = (fout*(2**16-1)).int().detach().cpu().numpy()[0,:,:,:]
            a = hsi_2.detach().cpu().numpy()[0,:,:,:]
            b = hsi_g.detach().cpu().numpy()[0,:,:,:]
            a = np.clip(a,0,1)
            temp =  []
            for i in range(31):
                # temp.append(skm.compare_psnr(hsi_g_[i,:,:],fout_[i,:,:],1))
                temp.append(skm.compare_psnr(b[i,:,:],a[i,:,:],1))
            temp = np.mean(np.array(temp))
            psnr_g.append(temp)
            # fout_0 = np.transpose(fout_,(0,2,3,1))))[0,:,:,:]
            # print(fout_0.shape)
            # print[0,:,:,:]
            # hsi_g_0 = np.transpose(hsi_g_,(0,2,3,1(hsi_g_0.shape)
            # ssim = skm.compare_ssim(X =fout_0*1.0/(2**16-1), Y =hsi_g_0*1.0/(2**16-1),K1 = 0.01, K2 = 0.03,multichannel=True)
            # ssim_log.append(ssim)
    # ssim_ = np.mean(np.array(ssim_log))
    psnr_ = np.mean(np.array(psnr_g))
    if 0:
        psnrl = torch.cat(psnrl,0)
        print('PSNR:{}'.format(psnrl.mean(-1).item()))
    else:
        # psnrl = np.array(psnrl)
        # psnrl1 = np.array(psnrl1)
        # psnrl2 = np.array(psnrl2)
        # RMSE = np.array(RMSE)
        # print('PSNRspa:{}'.format(psnrl1))
        # print('PSNRspe:{}'.format(psnrl2))
        print('PSNR:{}'.format(psnr_))
        # print('RMSE:{}'.format(RMSE.mean()))
