# coding=UTF-8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
#from tensorboardX import SummaryWriter

import os
import datetime
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import psutil
import numpy as np
import sys
import getopt
from clean_util import show_time, delta_time,save_checkpoint,load_checkpoint,grad_img,creat_P,para_setting,H_z,HT_y
from network_31 import VSR_CAS
from clean_dataset import ImageFolder
from tensorboardX import SummaryWriter
import scipy.io as sio
import util
from scipy.io import loadmat,savemat
# from util import  PSNR

#print(torch.cuda.is_available())
# Hyper Parameters
LR = 1e-4 # Learning Rate
WEIGHT_DECAY = 1e-8 # params of ADAM
delta  = 0.01
batch_size = 4 
patch_size = 96
channel = 31
checkpoint = 0
start_epoch = 0
EPOCH = 200000
task = 'SR'
delta = 2.0 #std of gaussian blur 
# h = 256
# w = 448
up_factor = 8  ######### Scale Factor #########
# task = '8_SR'
# save_path = './' + task +'/'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# --------------------------------------------------------------
# prepare DataLoader
HR_dir    =  '../data/CAVE/train/HSI'
RGB_dir   =  '../data/CAVE/train/RGB' # prepare the RGB pictures in advance using the reflectance matrix same with NSSR 
trainlist =  '../pathlist/'+'datalist_NSSR_P.txt'
# folders for saving training  results
date='0421_8_CAVE_Gaussian'
if not os.path.exists('./'+date):
    os.mkdir('./'+date)
model_path = './'+date+'/models'
if not os.path.exists(model_path):
    os.mkdir(model_path)
ckpt_path = './'+date+'/checkpoints'
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)
info_path = './'+date+'/model_train_info'
if not os.path.exists(info_path):
    os.mkdir(info_path)
# --------------------------------------------------------------
#  Prepare parameters for hyperspectral and spatial sampling

data = sio.loadmat('../data/P')
P    = data['P']
P    = torch.FloatTensor(P)
fft_B,fft_BT = para_setting('gaussian_blur', up_factor, [patch_size, patch_size], delta=delta)
fft_B = torch.cat( (torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)) ,2 )
fft_BT = torch.cat( (torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)) ,2 )
#-------------------------------------------------------------------
Dataset = ImageFolder(fft_B=fft_B,patch_size=patch_size, HR_dir=HR_dir, RGB_dir=RGB_dir,trainlist=trainlist, task=task,factor= up_factor,channel=channel,crop=True)
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=batch_size, num_workers=4,shuffle=True, drop_last=True) #num_workers=4,
sample_size = Dataset.count
print('================Total datasets number is : ',sample_size)

# --------------------------------------------------------------
vsr = torch.nn.DataParallel(VSR_CAS(channel0=channel , factor=up_factor , P=P ,patch_size =patch_size ).cuda())
print ("network is:",vsr)
optimizer = torch.optim.Adam(vsr.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_func = torch.nn.L1Loss()
# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []
start_epoch = 0
check_loss = 1
# if checkpoint!=0:
#     pretrained = torch.load(model_path + '/eopch_{}_params.pkl'.format(checkpoint))
#     vsr = torch.nn.DataParallel(VSR_CAS(channel0=channel , factor=up_factor , P=P ,patch_size =patch_size).cuda())
#     vsr.load_state_dict(pretrained)

writer = SummaryWriter(log_dir='./'+date+'/logs')
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000,10000,30000], 0.5)
psnr_best = 0
best_epoch = 0
for epoch in range(checkpoint, EPOCH):
    losses = []
    count = 0
    num_batch = sample_size //batch_size +1
    t0 = time.time()
    for step, (HR,LR , RGB) in enumerate(train_loader):
        HR = HR.cuda()
        LR = LR.cuda()
        RGB = RGB.cuda()
        #t1 = time.time()
        print("DataLoading time: {:.4f}".format(t1-t0))
        residual = vsr(LR,RGB)
        pred = residual 
        pow_num = num_batch * (epoch ) + (step+1)
        loss = (HR - pred).abs().mean()#+0.05*torch.acos(torch.clamp(torch.nn.functional.cosine_similarity(HR.permute(0,2,3,1),pred.permute(0,2,3,1),dim=-1),min=-1,max=1)).mean()
        #t2 = time.time()
        print("Forward time: {:.4f}".format(t2 - t1))
        losses.append(loss.item())
        writer.add_scalar('loss', loss.item(), pow_num - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t3 = time.time()
        #print("Backward time: {:.4f}".format(t3-t2))
        count += len(HR)
        print('%s  Processed %0.2f%% triples.' %
              (show_time(datetime.datetime.now()), float(count) / sample_size * 100))
        print('Epoch  %d ======Step  %d ============Loss1: %s \n' % (epoch, step ,loss.item()))
        t0 = time.time()
    #scheduler.step()
    ############################# validation ###################################    
    # LR_dir_val = './CAVE/NSSR_P/test/LR_8_gaussian'
    # HR_dir_val = './CAVE/NSSR_P/test/HSI'
    # RGB_dir_val = './CAVE/NSSR_P/test/RGB'
    # imgs_val = os.listdir(LR_dir_val)
    # x_axis = np.arange(1, 13)
    # if epoch % 100 ==0:
    #     vsr.module.patch_size = 512
    #     psnrs = []
    #     with torch.no_grad():
    #         for i in range(len(imgs_val)):
    #             print(imgs_val[i])
    #             LR = loadmat(os.path.join(LR_dir_val, imgs_val[i]))
    #             LR = torch.FloatTensor(LR['lr'])#.permute(2, 0, 1).unsqueeze(0).cuda()
    #             # noisy = noisy/255.0
    #             HR = loadmat(os.path.join(HR_dir_val, imgs_val[i]))
    #             HR = torch.FloatTensor(HR['hsi']).permute(2, 0, 1).unsqueeze(0).cuda()

    #             RGB = loadmat(os.path.join(RGB_dir_val, imgs_val[i]))
    #             RGB = torch.FloatTensor(RGB['rgb']).permute(2, 0, 1).unsqueeze(0).cuda()

    #             residual = vsr(LR, RGB)
    #             res = residual
    #             res = torch.clamp(res, 0, 1)
    #             psnr = PSNR(res, HR)
    #             print(psnr)
    #             psnrs.append(psnr)
    #             res = np.array(res.cpu())
    #             # save_path = './results/' + imgs_val[i]
    #             # savemat(save_path, {'res': res})

    #         psnrw_mean = sum(psnrs) / len(psnrs)
    #         print('The mean psnr is %f' % psnrw_mean)

    #         writer.add_scalar('psnr', psnrw_mean, epoch )
    #         if psnrw_mean > psnr_best:
    #             psnr_best = psnrw_mean
    #             torch.save(vsr.state_dict(), os.path.join(model_path, 'the_best_eopch_params.pkl'))
    #             save_checkpoint(vsr, optimizer, epoch + 1, ploty, ckpt_path + '/checkpoint_best.ckpt')
    #             plt.cla()
    #             plt.grid(linestyle='-.')
    #             for a, b in zip(x_axis, psnrs):
    #                 plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

    #             plt.plot(x_axis, psnrs, color='red', label='PSNR')
    #             plt.savefig('./'+date+ '/PSNR/psnr.png')
    #             best_epoch = epoch
    #         print('Epoch  {} :  The mean psnr is {}'.format(epoch, psnrw_mean))
    #         print('The best epoch is :{} ; The best psnr is {}'.format(best_epoch, psnr_best))   
    # ############################# validation ###################################
    #vsr.module.patch_size = 96
    if epoch % 100 ==0:
        torch.save(vsr.state_dict(),os.path.join(model_path, 'eopch_%d_params.pkl' % (epoch)))
        print('Saved.\n')
    print('\n%s  epoch %d: Average_loss=%f\n' % (show_time(datetime.datetime.now()), epoch , sum(losses)/ len(losses)))
    # checkpoint and then prepare for the next epoch
    if epoch % 5000 == 0:
        save_checkpoint(vsr, optimizer, epoch + 1, ploty, ckpt_path  + '/checkpoint_%d_epoch.ckpt' % (epoch + 1))
    t1 = time.time()

    if check_loss > sum(losses)/ len(losses):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        torch.save(vsr.state_dict(), os.path.join(model_path, 'vsr_models_eopch_%d_best_params.pkl' % (epoch)))
        print('Saved.\n')
        check_loss = sum(losses)/ len(losses)
    t2 = time.time()
    print(t2-t1)

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))
print('\n%s Saving model...' % show_time(datetime.datetime.now()))
# just save the parameters.
torch.save(vsr.state_dict(), os.path.join( model_path,'vsr_models_final_params.pkl'))
print('\n%s  Collecting some information...' % show_time(datetime.datetime.now()))
fp = open(os.path.join(info_path, 'model_grad_information_txt.txt'), 'w')
fp.write('Model Path:%s\n' % os.path.join(model_path, 'vsr_models_final_params.pkl'))
fp.write('\nModel Structure:\n')
fp.write('%s\n'%(vsr))
fp.write('\nModel Hyper Parameters:\n')
fp.write('\tEpoch = %d\n' % EPOCH)
fp.write('\tBatch size = %d\n' % batch_size)
fp.write('\tLearning rate = %f\n' % LR)
fp.write('Train on %dK  for %d epoch %s\n' % (int(sample_size), EPOCH ,'competition'))
fp.write("Training costs %02d:%02d:%02d" % (h, m, s) )
fp.close()
cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Totally costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))
print('%s  All done.' % show_time(datetime.datetime.now()))
