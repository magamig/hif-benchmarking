# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 20:08:46 2021

@author: 13572
"""




import torch
import torch.nn as nn
from evaluation import compute_sam,compute_psnr,compute_ergas,compute_cc,compute_rmse
import numpy as np
import random
from func import print_current_precision,print_options
import matplotlib.pyplot as plt

from model.spectral_upsample   import Spectral_upsample
from model.spectral_downsample import Spectral_downsample
from model.spatial_downsample import Spatial_downsample

from config import args
from Data_loader import Dataset
import time

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2)     #seed is set to 2


train_dataset=Dataset(args)
hsi_channels=train_dataset.hsi_channel
msi_channels=train_dataset.msi_channel
sp_range=train_dataset.sp_range
sp_matrix=train_dataset.sp_matrix
psf=train_dataset.PSF


#store the training configuration in opt.txt   
print_options(args)


lhsi=train_dataset[0]["lhsi"].unsqueeze(0).to(args.device) # change 3-order to 4-order(add batch)，i.e., from C,H,W to B,C,H,W (Meet the input requirements of pytorch)
hmsi=train_dataset[0]['hmsi'].unsqueeze(0).to(args.device)
hhsi=train_dataset[0]['hhsi'].unsqueeze(0).to(args.device)
lrmsi_frommsi=train_dataset[0]['lrmsi_frommsi'].unsqueeze(0).to(args.device)
lrmsi_fromlrhsi=train_dataset[0]['lrmsi_fromlrhsi'].unsqueeze(0).to(args.device)

#reference 3-order H,W,C
hhsi_true=hhsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_lrhsi_true=lhsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_msi_true=hmsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_frommsi_true= lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_fromlrhsi_true= lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1,2,0)
                
#generate three modules
Spectral_up_net   = Spectral_upsample(args,msi_channels,hsi_channels,init_type='normal', init_gain=0.02,initializer=False)
print('*******************************')
Spectral_down_net = Spectral_downsample(args, hsi_channels, msi_channels, sp_matrix, sp_range,  init_type='Gaussian', init_gain=0.02,initializer=True)
print('*******************************')
Spatial_down_net  = Spatial_downsample(args,psf, init_type='mean_space', init_gain=0.02,initializer=True)


optimizer_Spectral_down=torch.optim.Adam(Spectral_down_net.parameters(),lr=args.lr_stage1)
optimizer_Spatial_down=torch.optim.Adam(Spatial_down_net.parameters(),lr=args.lr_stage1)
optimizer_Spectral_up=torch.optim.Adam(Spectral_up_net.parameters(),lr=args.lr_stage2) 

L1Loss = nn.L1Loss(reduction='mean')
lr=args.lr_stage1

'''
#begin stage 1
'''
##S1 start
s1_start_time=time.time()
for epoch in range(1,args.epoch_stage1+1):
    
    optimizer_Spatial_down.zero_grad()
    optimizer_Spectral_down.zero_grad()
    
    out_lrmsi_fromlrhsi=Spectral_down_net(lhsi)  #spectrally degraded from lrhsi
    out_lrmsi_frommsi=Spatial_down_net(hmsi)     #spatially degraded from hrmsi
    loss1=L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)
        
        
    loss1.backward()
        
    optimizer_Spatial_down.step()
    optimizer_Spectral_down.step()
    
    if epoch % 100 ==0:  #print traning results in the screen every 100 epochs
            
            
            with torch.no_grad():
                out_fromlrhsi=out_lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1,2,0) #spectrally degraded from lrhsi
                out_frommsi  =out_lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1,2,0) #spatially degraded from hrmsi
                
                print('estimated PSF:',Spatial_down_net.psf.weight.data)
                print('true PSF:',psf)
                print('************')
                
                train_message='two generated images, train epoch:{} lr:{}\ntrain:L1loss:{}, sam_loss:{}, psnr:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_fromlrhsi-out_frommsi ) ) ,
                                 compute_sam(out_frommsi,out_fromlrhsi) ,
                                 compute_psnr(out_frommsi,out_fromlrhsi) ,
                                 compute_cc(out_frommsi,out_fromlrhsi),
                                 compute_rmse(out_frommsi,out_fromlrhsi)
                                 )
                print(train_message)
                
                print('************')
                test_message_SRF='SRF: generated lrmsifromlhsi and true lrmsifromlhsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_fromlrhsi-out_fromlrhsi_true ) ) ,
                                 compute_sam(out_fromlrhsi_true,out_fromlrhsi) ,
                                 compute_psnr(out_fromlrhsi_true,out_fromlrhsi) ,
                                 compute_ergas(out_fromlrhsi_true,out_fromlrhsi,args.scale_factor),
                                 compute_cc(out_fromlrhsi_true,out_fromlrhsi) ,
                                 compute_rmse(out_fromlrhsi_true,out_fromlrhsi)
                                 )
                print(test_message_SRF)
                
                print('************')
                test_message_PSF='PSF: generated lrmsifrommsi and true lrmsifrommsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_frommsi_true-out_frommsi ) ) ,
                                 compute_sam(out_frommsi_true,out_frommsi) ,
                                 compute_psnr(out_frommsi_true,out_frommsi) ,
                                 compute_ergas(out_frommsi_true,out_frommsi,args.scale_factor),
                                 compute_cc(out_frommsi_true,out_frommsi),
                                 compute_rmse(out_frommsi_true,out_frommsi)
                                 )
                print(test_message_PSF)
                print('\n')
                
    if (epoch>args.decay_begin_epoch_stage1-1): 
                each_decay=args.lr_stage1/(args.epoch_stage1-args.decay_begin_epoch_stage1+1)
                lr = lr-each_decay
                for param_group in optimizer_Spectral_down.param_groups:
                    param_group['lr'] = lr 
                for param_group in optimizer_Spatial_down.param_groups:
                    param_group['lr'] = lr 

###store the result in Stage 1
print_current_precision(args,'results in Stage 1')
print_current_precision(args,train_message)
print_current_precision(args,test_message_SRF)
print_current_precision(args,test_message_PSF)
print_current_precision(args,'estimated PSF:\n{}'.format(Spatial_down_net.psf.weight.data))
print_current_precision(args,'true PSF:\n{}'.format(psf))

temp1=[Spectral_down_net.conv2d_list[i].weight.data.cpu().numpy()[0,:,0,0] for i in range(0,sp_range.shape[0])]
temp2=[temp1[i].sum() for i in range(0,sp_range.shape[0])]
estimated_SRF=[temp1[i]/temp2[i] for i in range(0,sp_range.shape[0])]
print_current_precision(args,'estimated SRF:\n{}'.format(estimated_SRF))
print_current_precision(args,'true SRF:\n{}'.format([sp_matrix[int(sp_range[i,0]):int(sp_range[i,1])+1,i] for i in range(0,sp_range.shape[0])]))
###

'''
#begin stage 2
'''
out_lrmsi_frommsi_new=out_lrmsi_frommsi.clone().detach()                
print('____________________________________________________')

lr=args.lr_stage2
for epoch in range(1,args.epoch_stage2+1):       
    optimizer_Spectral_up.zero_grad()
    lrhsi=Spectral_up_net(out_lrmsi_frommsi_new)    #learn SpeUnet, the spectral inverse mapping from low MSI to low HSI
    loss2=L1Loss(lrhsi, lhsi)  
    loss2.backward()
    optimizer_Spectral_up.step()
    if epoch % 100 ==0:  #print traning results in the screen every 100 epochs

            with torch.no_grad():
                out_lrhsi=lrhsi.detach().cpu().numpy()[0].transpose(1,2,0)       
                
                est_hhsi=Spectral_up_net(hmsi).detach().cpu().numpy()[0].transpose(1,2,0) #use the learned SpeUnet to generate estimated HHSI in the second stage
                
                train_message_specUp='genrated lrhsi and true lrhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_lrhsi_true-out_lrhsi ) ) ,
                                 compute_sam(out_lrhsi_true,out_lrhsi) ,
                                 compute_psnr(out_lrhsi_true,out_lrhsi) ,
                                 compute_ergas(out_lrhsi_true,out_lrhsi,args.scale_factor) ,
                                 compute_cc(out_lrhsi_true,out_lrhsi),
                                 compute_rmse(out_lrhsi_true,out_lrhsi)
                                 )
                print(train_message_specUp)
                print('************')
                
                test_message_specUp='generated hhsi and true hhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( hhsi_true-est_hhsi ) ) ,
                                 compute_sam(hhsi_true,est_hhsi) ,
                                 compute_psnr(hhsi_true,est_hhsi) ,
                                 compute_ergas(hhsi_true,est_hhsi,args.scale_factor),
                                 compute_cc(hhsi_true,est_hhsi),
                                 compute_rmse(hhsi_true,est_hhsi)
                                 )
                print(test_message_specUp)
                print('\n')
                
    if (epoch>args.decay_begin_epoch_stage2-1): 
                each_decay=args.lr_stage2/(args.epoch_stage2-args.decay_begin_epoch_stage2+1)
                lr = lr-each_decay
                for param_group in optimizer_Spectral_up.param_groups:
                    param_group['lr'] = lr 

##S2 end
s1s2_over_time=time.time()
s1s2_time=s1s2_over_time-s1_start_time

###store the result in Stage 2
print_current_precision(args,'\n')
print_current_precision(args,'results in Stage 2')
print_current_precision(args,train_message_specUp)
print_current_precision(args,test_message_specUp)
print_current_precision(args,'time of S1+S2:{}'.format(s1s2_time))
#########################


'''
#begin stage 3
'''
print('____________________________________________________')
#
for param_group in optimizer_Spectral_down.param_groups:
    param_group['lr'] = args.lr_stage3
for param_group in optimizer_Spatial_down.param_groups:
    param_group['lr'] = args.lr_stage3
for param_group in optimizer_Spectral_up.param_groups:
    param_group['lr'] = args.lr_stage3 
lr=args.lr_stage3 
for epoch in  range(1,args.epoch_stage3+1):       
        optimizer_Spectral_up.zero_grad()
        optimizer_Spatial_down.zero_grad()
        optimizer_Spectral_down.zero_grad()
        #moduleⅠ 和 Ⅱ
        out_lrmsi_fromlrhsi=Spectral_down_net(lhsi)   #spectrally degraded from lrhsi
        out_lrmsi_frommsi=Spatial_down_net(hmsi)       #spatially degraded from hrmsi
        loss1=L1Loss(out_lrmsi_fromlrhsi, out_lrmsi_frommsi)
        
        #module Ⅲ
        lrhsi=Spectral_up_net(out_lrmsi_frommsi)    #learn SpeUnet, the spectral inverse mapping from low MSI to low HSI
        loss2=L1Loss(lrhsi, lhsi)                   
        
        pre_hhsi=Spectral_up_net(hmsi)     #use the learned SpeUnet to generate estimated HHSI in the last stage
        pre_msi=Spectral_down_net(pre_hhsi)  #spectrally degraded from pre_hhsi
        pre_lrhsi=Spatial_down_net(pre_hhsi)   #spatially degraded from pre_hhsi
        
        loss3=L1Loss(pre_msi,hmsi)
        loss4=L1Loss(pre_lrhsi,lhsi)
        
        
        loss=loss1+loss2+loss3+loss4  
       

        loss.backward()
        
        optimizer_Spectral_up.step()
        optimizer_Spatial_down.step()
        optimizer_Spectral_down.step()

        if epoch % 100 ==0:   #print traning results in the screen every 100 epochs
            with torch.no_grad():

                out_fromlrhsi=out_lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1,2,0) #spectrally degraded from lrhsi
                out_frommsi  =out_lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1,2,0) #spatially degraded from hrmsi
                
                out_lrhsi=lrhsi.detach().cpu().numpy()[0].transpose(1,2,0)        #the spectral inverse mapping from low MSI to low HSI
                
                est_hhsi=Spectral_up_net(hmsi).detach().cpu().numpy()[0].transpose(1,2,0) #use the learned SpeUnet to generate estimated HHSI 
               
                
                out_pre_msi  =pre_msi.detach().cpu().numpy()[0].transpose(1,2,0)  #spectrally degraded from pre_hhsi
                out_pre_lrhsi=pre_lrhsi.detach().cpu().numpy()[0].transpose(1,2,0) #spatially degraded from pre_hhsi
                
                
                print('estimated PSF:',Spatial_down_net.psf.weight.data)
                print('true PSF:',psf)
                print('************')
                
                train_message='two generated images, train epoch:{} lr:{}\ntrain:L1loss:{}, sam_loss:{}, psnr:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_fromlrhsi-out_frommsi ) ) ,
                                 compute_sam(out_frommsi,out_fromlrhsi) ,
                                 compute_psnr(out_frommsi,out_fromlrhsi) ,
                                 compute_cc(out_frommsi,out_fromlrhsi),
                                 compute_rmse(out_frommsi,out_fromlrhsi)
                                 )
                print(train_message)
                print('************')
                
                test_message_SRF='SRF: generated lrmsifromlhsi and true lrmsifromlhsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_fromlrhsi-out_fromlrhsi_true ) ) ,
                                 compute_sam(out_fromlrhsi_true,out_fromlrhsi) ,
                                 compute_psnr(out_fromlrhsi_true,out_fromlrhsi) ,
                                 compute_ergas(out_fromlrhsi_true,out_fromlrhsi,args.scale_factor),
                                 compute_cc(out_fromlrhsi_true,out_fromlrhsi),
                                 compute_rmse(out_fromlrhsi_true,out_fromlrhsi)
                                 )
                print(test_message_SRF)
                print('************')
                
                test_message_PSF='PSF: generated lrmsifrommsi and true lrmsifrommsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_frommsi_true-out_frommsi ) ) ,
                                 compute_sam(out_frommsi_true,out_frommsi) ,
                                 compute_psnr(out_frommsi_true,out_frommsi) ,
                                 compute_ergas(out_frommsi_true,out_frommsi,args.scale_factor),
                                 compute_cc(out_frommsi_true,out_frommsi),
                                 compute_rmse(out_frommsi_true,out_frommsi)
                                 )
                print(test_message_PSF)
                print('************')
                
                train_message_specUp='genrated lrhsi via SpeUnet and true lrhsi. epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_lrhsi_true-out_lrhsi ) ) ,
                                 compute_sam(out_lrhsi_true,out_lrhsi) ,
                                 compute_psnr(out_lrhsi_true,out_lrhsi) ,
                                 compute_ergas(out_lrhsi_true,out_lrhsi,args.scale_factor),
                                 compute_cc(out_lrhsi_true,out_lrhsi),
                                 compute_rmse(out_lrhsi_true,out_lrhsi)
                                 )
                print(train_message_specUp)
                print('************')
                
                train_message_spatial_down='generated lrhsi via SpaDnet from preHHSI and true lrhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs(out_lrhsi_true-out_pre_lrhsi ) ) ,
                                 compute_sam(out_lrhsi_true,out_pre_lrhsi) ,
                                 compute_psnr(out_lrhsi_true,out_pre_lrhsi) ,
                                 compute_ergas(out_lrhsi_true,out_pre_lrhsi,args.scale_factor),
                                 compute_cc(out_lrhsi_true,out_pre_lrhsi),
                                 compute_rmse(out_lrhsi_true,out_pre_lrhsi)
                                 )
                print(train_message_spatial_down)
                print('************')
                
                train_message_spectral_down='generated hrMSI via SpeDnet from preHHSI and true hrMSI epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( out_msi_true- out_pre_msi ) ) ,
                                 compute_sam(out_msi_true, out_pre_msi) ,
                                 compute_psnr(out_msi_true, out_pre_msi) ,
                                 compute_ergas(out_msi_true, out_pre_msi,args.scale_factor),
                                 compute_cc(out_msi_true, out_pre_msi),
                                 compute_rmse(out_msi_true, out_pre_msi)
                                 )
                          
                print(train_message_spectral_down)
                print('************')
                
                test_message_specUp='generated HHSI and true hhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}, dataname{}'.\
                          format(epoch,lr,
                                 np.mean( np.abs( hhsi_true-est_hhsi ) ) ,
                                 compute_sam(hhsi_true,est_hhsi) ,
                                 compute_psnr(hhsi_true,est_hhsi) ,
                                 compute_ergas(hhsi_true,est_hhsi,args.scale_factor),
                                 compute_cc(hhsi_true,est_hhsi),
                                 compute_rmse(hhsi_true,est_hhsi),
                                 args.data_name
                                 )
                print(test_message_specUp)
                
                print('\n')
        

        if (epoch>args.decay_begin_epoch_stage3-1): 
                each_decay=args.lr_stage3/(args.epoch_stage3-args.decay_begin_epoch_stage3+1)
                lr = lr-each_decay
                for param_group in optimizer_Spectral_down.param_groups:
                    param_group['lr'] = lr 
                for param_group in optimizer_Spatial_down.param_groups:
                    param_group['lr'] = lr 
                for param_group in optimizer_Spectral_up.param_groups:
                    param_group['lr'] = lr     
#S3 end
s3_over_time=time.time()
s3_time=s3_over_time-s1_start_time

###store the result in last Stage 
print_current_precision(args,'\n')
print_current_precision(args,'results in Stage 3')

print_current_precision(args,train_message)
print_current_precision(args,test_message_PSF)
print_current_precision(args,test_message_SRF)

print_current_precision(args,'estimated PSF:\n{}'.format(Spatial_down_net.psf.weight.data))
print_current_precision(args,'true PSF:\n{}'.format(psf))

temp1=[Spectral_down_net.conv2d_list[i].weight.data.cpu().numpy()[0,:,0,0] for i in range(0,sp_range.shape[0])]
temp2=[temp1[i].sum() for i in range(0,sp_range.shape[0])]
estimated_SRF=[temp1[i]/temp2[i] for i in range(0,sp_range.shape[0])]
print_current_precision(args,'estimated SRF:\n{}'.format(estimated_SRF))
print_current_precision(args,'true SRF:\n{}'.format([sp_matrix[int(sp_range[i,0]):int(sp_range[i,1])+1,i] for i in range(0,sp_range.shape[0])]))

print_current_precision(args,train_message_specUp)
print_current_precision(args,train_message_spatial_down)
print_current_precision(args,train_message_spectral_down)
print_current_precision(args,test_message_specUp)
print_current_precision(args,'time of S3:{}'.format(s3_time))



from func import save_net
##save trained three module
save_net(args,Spectral_up_net)
save_net(args,Spectral_down_net)
save_net(args,Spatial_down_net)


###save estimated HHSI
from func import save_hhsi
save_hhsi(args,est_hhsi)
print(args)
print('all done')





