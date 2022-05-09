import torch
from SpaNet import SpaNet
from SpeNet import SpeNet,recon_net
import argparse
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import itertools
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
import visdom
from torch.utils import data
from Hyper_loader_2 import Hyper_dataset
from torchvision.utils import make_grid
import numpy as np
from SSIM import SSIM
# from apex import amp
import os
import time
from torch.utils.data import DataLoader
import skimage.measure as skm
import skimage as ski
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
port = 8100
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
def xavier_init_(model):
    for name,param in model.named_parameters():
        if 'weight' in name:
            x_init(param)

def loss_spa(hsi,hsi_t,Mse):
    loss = Mse(hsi[1],hsi_t[1]) + Mse(hsi[2],hsi_t[2])
    return loss
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

def save_model(epoch,Renet):
    model_state_dict = {'spenet':Renet.state_dict()}
    os.makedirs('./save_model'+now,exist_ok=True)
    torch.save(model_state_dict,'./save_model'+now+'/state_dicr_{}.pkl'.format(epoch))

def load_model(model,mode_dict):
    # state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in mode_dict.items():
        # name = k[7:] # remove `module.`
        name = k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
def get_spe_gt(hsi):
    output=[]
    output.append(hsi)
    index = [np.array(list(range(8)))*4,np.array(list(range(16)))*2]
    index[0][-1] = 30
    index[1][-1] = 30
    output.append(hsi[:,index[0],:,:])
    output.append(hsi[:,index[1],:,:])
    return output
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
conti = True
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_epochs', type = int, default = 1000, help = 'number of epochs of training')
    parse.add_argument('--batch_size',type = int, default = 6, help = 'size of the batches')
    parse.add_argument('--lr',type = float, default = 1e-3, help='learing rate of network')
    parse.add_argument('--b1',type = float, default = 0.9, help='adam:decay of first order momentum of gradient')
    parse.add_argument('--b2',type = float, default = 0.999, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--envname',type = str, default = 'temp', help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--local_rank', type=int, default=0)
    setup_seed(50)
    factor_lr = 1
    # spanet = SpaNet().cuda()
    opt = parse.parse_args()
    torch.cuda.set_device(opt.local_rank)
    Hyper_test = Hyper_dataset(output_shape=128,Training_mode='Test',data_name='CAVE')
    # Hyper_test_sampler  = torch.utils.data.distributed.DistributedSampler(Hyper_test, shuffle=False)
    Hyper_test = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=4,pin_memory=True,
                                     drop_last=True)

    Hyper_train = Hyper_dataset(output_shape=128,Training_mode='Train',data_name='CAVE')
    datalen = Hyper_train.__len__()
    # Hyper_train_sampler  = torch.utils.data.distributed.DistributedSampler(Hyper_train, shuffle=True)
    Hyper_train = DataLoader(Hyper_train,batch_size=opt.batch_size,shuffle=False, num_workers=10,pin_memory=True,
                                     drop_last=True)
    device = torch.device('cuda:{}'.format(opt.local_rank))
    spenet = SpeNet().cuda().to(device)
    # renet = recon_net(cin=31).cuda().to(device)
    opt = parse.parse_args()
    optimzier = torch.optim.Adam(itertools.chain(spenet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    # MseLoss = torch.nn.MSELoss()
    MaeLoss = torch.nn.L1Loss().to(device)   
    T_max = (datalen//(1*opt.batch_size))*1000
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzier, T_max, eta_min=1e-5, last_epoch=-1) 
    env=opt.envname
    server='10.37.0.18'
    Loss_logger = VisdomPlotLogger('line',opts={'title':'loss2'},port=8100,env=env,server=server)
    Losst_logger = VisdomPlotLogger('line',opts={'title':'loss_t2'},port=8100,env=env,server=server)
    Lossm_logger = VisdomPlotLogger('line',opts={'title':'SSIM_t2'},port=8100,env=env,server=server)
    PSNR_logger = VisdomPlotLogger('line',opts={'title':'PSNR'},port=8100,env=env,server=server)
    train_logger = VisdomLogger('image',opts= {'title':'reconstruction image2'},port=8100,env=env,server=server)
    test_logger = VisdomLogger('image',opts={'title':'Residual image2'},port=8100,env=env,server=server)
    b_r = 0
    ssim_best = 0
    psnr_best = 0
    if conti == True:
        state_dict = torch.load('/home/zhu_19/data/saperate/run_folders/2020-04-05-15-29-38/save_model2020-04-05-15-29-40/state_dicr_500.pkl')
        del state_dict
    for epoch in range(opt.n_epochs):
        batch = 0
        loss_ = []
        loss_t = []
        ssim_log = []
        for hsi,hsi_g,hsi_resize,msi in Hyper_train:
            batch = batch+1
            b_r = max(batch,b_r)
            spenet.train()
            # if conti == True:
            #     freeze_bn(spenet)
            #     if epoch <300:
            #         adjust_learning_rate(optimzier,lr = 1e-3)
            #     else:
            #         adjust_learning_rate(optimzier,lr = 1e-4)
            # else:
            #     if epoch > 500:
            #         freeze_bn(spenet)
            #     if epoch > 900:
            #         adjust_learning_rate(optimzier,lr = 1e-4)
            hsi_resize = hsi_resize.cuda().float()
            hsi_resize = torch.nn.functional.interpolate(hsi_resize,scale_factor=(8,8),mode='bilinear')
            # with torch.no_grad():
            hsi_g = hsi_g.cuda().float()
            # hsi = renet(hsi_resize)
            hsi = hsi_resize
            # hsi = hsi.cuda()
            scale_g = nn.AdaptiveAvgPool2d(1)(hsi_g)
            msi = [a.cuda().float().to(device) for a in msi]
            hsi_spe,scale,refined = spenet(hsi.to(device),msi[-1])
            scale_a = nn.AdaptiveAvgPool2d(1)(scale)
            hsi_2 = hsi_spe[-1][:,:31,:,:]
            # fout = hsi_2 + scale_a
            fout = hsi_2 + hsi
            optimzier.zero_grad()
            hsi_g_ = hsi_g - hsi

            loss = MaeLoss(fout.to(device),hsi_g.to(device))+MaeLoss(refined.to(device),hsi_g.to(device))
            
            loss.backward()
            optimzier.step()
            schduler.step()
            loss_.append(loss.item())
            del loss
            if opt.local_rank == 0:
                print('[Epoch:{}/{}][batch:{}/{}][loss:{}][learning rate:{}]'.format(epoch,opt.n_epochs,batch,b_r,loss_[-1],optimzier.state_dict()['param_groups'][0]['lr']))
        a = np.random.randint(0,31,fout.shape[0])
        a_ = np.array(range(fout.shape[0]))
        output = fout.detach().cpu()[a_,a,:,:]*255
        output = output[:,None,:,:]
        output = torch.cat([output,output,output],1)
        output_ = make_grid(output,nrow=int(5)).numpy()
        if opt.local_rank == 0:
            train_logger.log(output_)
            Loss_logger.log(epoch,np.mean(np.array(loss_)))
        psnr_g = []
        with torch.no_grad():
            for hsi,hsi_g,hsi_resize,msi in Hyper_test:
                spenet.eval()
                hsi_g = hsi_g.cuda().float()
                hsi_resize = hsi_resize.cuda().float()
                hsi_resize = torch.nn.functional.interpolate(hsi_resize,scale_factor=(8,8),mode='bilinear')
                # hsi = renet(hsi_resize)
                hsi = hsi_resize
                # hsi = [a.cuda().float() for a in hsi]
                msi = [a.cuda().float() for a in msi]
                hsi_spe,scale,refined  = spenet(hsi,msi[-1])
                # hsi_e = get_spe_gt(hsi_g)
                # hsi_2 = hsi_spe[-1][:,:31,:,:]
                fout = refined
                # loss = MaeLoss(fout,hsi_g)
                # loss_t.append(loss.item())
                hsi_g_ = (hsi_g*(2**16-1)).int().detach().cpu().numpy()
                fout_ = (fout*(2**16-1)).int().detach().cpu().numpy()
                for i in range(31):
                    psnr_g.append(skm.compare_psnr(hsi_g_[0,i,:,:],fout_[0,i,:,:],2**16-1))
                fout_0 = np.transpose(fout_,(0,2,3,1))[0,:,:,:]
                hsi_g_0 = np.transpose(hsi_g_,(0,2,3,1))[0,:,:,:]
                ssim = skm.compare_ssim(X =fout_0*1.0/(2**16-1), Y =hsi_g_0*1.0/(2**16-1),K1 = 0.01, K2 = 0.03,multichannel=True)
                ssim_log.append(ssim)
        ssim_ = np.mean(np.array(ssim_log))
        psnr_ = np.mean(np.array(psnr_g))
        if opt.local_rank == 0:
            PSNR_logger.log(epoch,psnr_)
            Losst_logger.log(epoch,np.mean(np.array(loss_t)))
            Lossm_logger.log(epoch,ssim_)
        if ssim_best < ssim_ and epoch > 900 and opt.local_rank == 0:
            ssim_best = ssim_
            save_model(epoch,spenet)
        if psnr_best < psnr_ and epoch > 900 and opt.local_rank == 0:
            psnr_best = psnr_
            save_model(epoch,spenet)
        a = np.random.randint(0,31,fout.shape[0])
        a_ = np.array(range(fout.shape[0]))
        output = fout.detach().cpu()[a_,a,:,:]*255
        output = output[:,None,:,:]
        output = torch.cat([output,output,output],1)
        output_ = make_grid(output,nrow=int(5)).numpy()
        if opt.local_rank == 0:
            test_logger.log(output_)
            if epoch % 100 == 0:
                save_model(epoch,spenet)
    if opt.local_rank == 0:
        save_model(epoch,spenet)