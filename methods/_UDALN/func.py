# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import scipy.io as sio

'''
some functions used in train_all_special.py
'''

#store the precision result in precesion.txt
def print_current_precision(opt,message):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.data_name+'_'+str(opt.scale_factor)+'_S1={}_S2={}_S3={}'\
                .format( 
                         str(opt.lr_stage1)+'_'+str(opt.epoch_stage1)+'_'+str(opt.decay_begin_epoch_stage1),
                         str(opt.lr_stage2)+'_'+str(opt.epoch_stage2)+'_'+str(opt.decay_begin_epoch_stage2),
                         str(opt.lr_stage3)+'_'+str(opt.epoch_stage3)+'_'+str(opt.decay_begin_epoch_stage3)
                        
                       )
                )
    precision_path = os.path.join(expr_dir  , 'precision.txt')
    with open(precision_path, "a") as precision_file:
            precision_file.write('%s\n' % message)
            
#store the training configuration in opt.txt      
def print_options(opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.data_name+'_'+str(opt.scale_factor)+'_S1={}_S2={}_S3={}'\
                .format( 
                         str(opt.lr_stage1)+'_'+str(opt.epoch_stage1)+'_'+str(opt.decay_begin_epoch_stage1),
                         str(opt.lr_stage2)+'_'+str(opt.epoch_stage2)+'_'+str(opt.decay_begin_epoch_stage2),
                         str(opt.lr_stage3)+'_'+str(opt.epoch_stage3)+'_'+str(opt.decay_begin_epoch_stage3)
                        
                       )
                )
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
            
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

#save the trained three modules:1.convolution_hr2msi.pth  2.PSF.pth 3.spectral_upsample.pth
def save_net(opt,net):
    
    expr_dir = os.path.join(opt.checkpoints_dir, opt.data_name+'_'+str(opt.scale_factor)+'_S1={}_S2={}_S3={}'\
                .format( 
                         str(opt.lr_stage1)+'_'+str(opt.epoch_stage1)+'_'+str(opt.decay_begin_epoch_stage1),
                         str(opt.lr_stage2)+'_'+str(opt.epoch_stage2)+'_'+str(opt.decay_begin_epoch_stage2),
                         str(opt.lr_stage3)+'_'+str(opt.epoch_stage3)+'_'+str(opt.decay_begin_epoch_stage3)
                        
                       )
                )
    PATH=os.path.join(expr_dir,net.__class__.__name__+'.pth')
    torch.save(net.state_dict(),PATH)

#save the final result: My_Out.mat
def save_hhsi(opt,hhsi):
    
    expr_dir = os.path.join(opt.checkpoints_dir, opt.data_name+'_'+str(opt.scale_factor)+'_S1={}_S2={}_S3={}'\
                .format( 
                         str(opt.lr_stage1)+'_'+str(opt.epoch_stage1)+'_'+str(opt.decay_begin_epoch_stage1),
                         str(opt.lr_stage2)+'_'+str(opt.epoch_stage2)+'_'+str(opt.decay_begin_epoch_stage2),
                         str(opt.lr_stage3)+'_'+str(opt.epoch_stage3)+'_'+str(opt.decay_begin_epoch_stage3)
                        
                       )
                )
    PATH=os.path.join(expr_dir,'My_Out.mat')
    sio.savemat(PATH,{'Out':hhsi})
    
#obtain the coverage index between multispectral spectral response and hyperspectral wavelength
def get_sp_range(sp_matrix):
        HSI_bands, MSI_bands = sp_matrix.shape
    
        assert(HSI_bands>MSI_bands)
        sp_range = np.zeros([MSI_bands,2])
        for i in range(0,MSI_bands):
            index_dim_0, index_dim_1 = np.where(sp_matrix[:,i].reshape(-1,1)>0)
            sp_range[i,0] = index_dim_0[0] 
            sp_range[i,1] = index_dim_0[-1]
        return sp_range