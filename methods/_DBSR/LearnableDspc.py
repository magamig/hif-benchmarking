import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from downsampler import *
import math
import numpy as np
import scipy.io as sio
from model import *
from model_1 import *
from scipy.misc import imresize
import time
import copy
import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from DownKernel import *

##parameter setting
#use cuda, or not?
use_cuda = torch.cuda.is_available()
num_steps = 5001
#output_name = 'output_'
sigma = 1./30
#save_path = './output/Harvard_16_result/'
Dsets_name = 'WO_K3'
save_path = './WO_pre/a/8K3/'
dir_data = './DSpa_8K3/'
KT_pretrain = 'gauss12'   #kernel type of pretrain 'bilinear' or 'gauss12'
factor = 8


#get the spectual downsample func
def create_P():
    P_ = [[2,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
          [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2,  2,  1,  1,  2,  2,  2,  2,  2],
          [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]
    P = np.array(P_, dtype=np.float32)
    div_t = np.sum(P, axis=1)
    for i in range(3):
        P[i,] = P[i,]/div_t[i]
    return P

#get PSNR
def PSNR_GPU(im_true, im_fake):
    data_range = 1
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone().resize_(C*H*W)
    Ifake = im_fake.clone().resize_(C*H*W)
    #mse = nn.MSELoss(reduce=False)
    err = torch.pow(Itrue-Ifake,2).sum(dim=0, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)


#get SAM
def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone().resize_(C, H*W)
    Ifake = im_fake.clone().resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0).resize_(H*W)
    denom1 = torch.pow(Itrue,2).sum(dim=0).sqrt_().resize_(H*W)
    denom2 = torch.pow(Ifake,2).sum(dim=0).sqrt_().resize_(H*W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(H*W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (H*W)
    return sam

class RGB_sets(data.Dataset):
    def __init__(self,RGB,Label,P_N,P_S,factor):
        self.RGB = RGB
        self.size = RGB.shape
        self.P_S = P_S
        self.Label = Label
        self.P_N = P_N
        self.D_spa = Downsampler(n_planes=31,factor=factor,kernel_type=KT_pretrain,phase=0,preserve_size=True).type(torch.cuda.FloatTensor)

    def __getitem__(self, index):
        [p,q] = self.Label[index,:]
        P_S = self.P_S
        RGB = self.RGB[:,p:p+P_S,q:q+P_S]
        D_RGB = self.D_spa(torch.unsqueeze(RGB,0))

        return torch.squeeze(D_RGB,0),RGB

    def __len__(self):
        return self.P_N


class HSI_sets(data.Dataset):
    def __init__(self,HSI,Label,P_N,P_S):
        self.HSI = HSI
        self.size = HSI.shape
        self.P_S = P_S
        self.Label = Label
        self.P_N = P_N
        self.P = create_P()

    def __getitem__(self, index):
        [p,q] = self.Label[index,:]
        P_S = self.P_S
        HSI = self.HSI[:,p:p+P_S,q:q+P_S]
        P = torch.from_numpy(self.P).cuda()
        HSI = torch.reshape(HSI,(HSI.shape[0],P_S**2))
        D_HSI = torch.matmul(P,HSI).view(3,P_S,P_S)
        HSI = HSI.view(HSI.shape[0],P_S,P_S)

        return D_HSI,HSI

    def __len__(self):
        return self.P_N


def Get_Label(im_size, patch_size):
    m, n = 0, 0
    P_number = 0
    Lable_table = []
    while 1:
        if m + patch_size < im_size[1] and n + patch_size < im_size[0]:
            Lable_table.append([m, n])
            m = m + patch_size
            P_number += 1
        elif m + patch_size >= im_size[1] and n + patch_size < im_size[0]:
            m = im_size[1] - patch_size
            Lable_table.append([m, n])
            m, n = 0, n + patch_size
            P_number += 1
        elif m + patch_size < im_size[1] and n + patch_size >= im_size[0]:
            Lable_table.append([m, im_size[0] - patch_size])
            m += patch_size
            P_number += 1
        elif m + patch_size >= im_size[1] and n + patch_size >= im_size[0]:
            Lable_table.append([im_size[1] - patch_size, im_size[0] - patch_size])
            P_number += 1
            break
    return np.array(Lable_table), P_number


#Spatial upsample model
def Spa_UpNet(image):
    P_S = 64
    Label, P_n = Get_Label(image.shape[1:], P_S)
    dataset = RGB_sets(image, Label, P_n, P_S,factor=factor)
    data = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    N1 = nn.Sequential()
    N1.add(get_net_1(31,'skip','reflection',n_channels=31,skip_n33d=40,skip_n33u=40,skip_n11=1,num_scales=2,upsample_mode='bilinear'))
    N1.add(nn.Upsample(scale_factor=factor,mode='bilinear'))
    N1.add(get_net_1(31,'skip','reflection',n_channels=31,skip_n33d=40,skip_n33u=40,skip_n11=1,num_scales=2,upsample_mode='bilinear'))
    N1 = N1.cuda()
    L1Loss = nn.L1Loss(reduce=True)
    optimizer_N1 = torch.optim.Adam(N1.parameters(), lr=1e-3)

    #f1 = open(save_path+'spa_loss.txt','w') 

    for epoch in range(1500):
        running_loss = 0
        for i,batch in enumerate(data,1):
            lr,hr = batch[0],batch[1]
            lr,hr = Variable(lr).cuda(),Variable(hr).cuda()

            out = N1(lr)
            loss = L1Loss(out,hr)
            running_loss += loss.data[0]
            optimizer_N1.zero_grad()
            loss.backward()
            optimizer_N1.step()
        #f1.write('{}\n'.format(running_loss))
        print('*'*10,'At Epoch {},the loss is : {}.'.format(epoch,running_loss))
    #f1.close()
    return N1

#spectral upsample model
def Spc_UpNet(image):
    P_S = 16
    Label,P_n = Get_Label(image.shape[1:],P_S)
    dataset = HSI_sets(image,Label,P_n,P_S)
    data = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    N2 = get_net_1(3,'skip','reflection',n_channels=31,skip_n33d=31,skip_n33u=31,skip_n11=4,num_scales=5,upsample_mode='bilinear')
    N2 = N2.cuda()
    L1Loss = nn.L1Loss(reduce=True)
    optimizer_N2 = torch.optim.Adam(N2.parameters(), lr=1e-3)
    
    #f2 = open(save_path+'spc_loss.txt','w')
    
    for epoch in range(3000):
        running_loss = 0
        for i,batch in enumerate(data,1):
            lr,hr = batch[0],batch[1]
            lr,hr = Variable(lr).cuda(),Variable(hr).cuda()

            out = N2(lr)
            #print(out.shape,hr.shape)
            loss = L1Loss(out,hr)
            running_loss += loss.data[0]
            optimizer_N2.zero_grad()
            loss.backward()
            optimizer_N2.step()
        #f2.write('{}\n'.format(running_loss))
        #print('*'*10,'At Epoch {},the loss is : {}.'.format(epoch,running_loss))
    #f2.close()
    return N2

#pretrain a spectral downsample
def PreT_DspcNet (net,im_m,im_h):
    D_spa = Downsampler(n_planes=3,factor=factor,kernel_type='gauss12',phase=0,preserve_size=True).type(torch.cuda.FloatTensor)
    D_RGB = D_spa(torch.unsqueeze(im_m,0))
    D_RGB = Variable(D_RGB,requires_grad = False)
    Input = torch.unsqueeze(im_h,0)
    Mse = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    running_loss = 0

    for epoch in range(1000):
        out = net(Input)

        loss = Mse(out,D_RGB)
        running_loss += loss.data[0]
        if epoch%100 == 0:
            #print('*'*10,'At epoch {} the loss is : {}.'.format(epoch,running_loss))
            running_loss = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net

# main processing
if __name__ =='__main__':
    #dir_data = './Harvard_512'
    #dir_data = './Mydata'


    #define the lossfunction & optimizer
    mse = nn.MSELoss()

    k = 0
    files = os.listdir(dir_data)

    for names in files[0:16]:
    #for i in range(1):
        lr = 1e-3
        k += 1
        data = sio.loadmat(dir_data+names)
        #data = sio.loadmat('./beads_ms.mat')
        #data = sio.loadmat('./Mydata/beads_3.mat')
        name = names[:-6]
        #name = 'beads_ms'
        print('Producing with the {} image'.format(name))

        #im_gt = data['data']
        im_gt = data['data']

        #get_input
        net = get_net(im_gt.shape[0],'skip','reflection',n_channels=im_gt.shape[0],skip_n33d=256,skip_n33u=256,skip_n11=1,num_scales=2,upsample_mode='bilinear')
        
        if use_cuda:
            net = net.cuda()

        optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        #down_spa = Downsampler(n_planes=im_gt.shape[0],factor=factor,kernel_type='gauss12',phase=0,preserve_size=True).type(torch.cuda.FloatTensor)

        # the fixed spectral downsampler
        p = create_P()
        p = Variable(torch.from_numpy(p.copy()), requires_grad=False).cuda()

        net_input = np.zeros(im_gt.shape, dtype=np.float32)
        #trans the data to Variable
        if use_cuda:
            im_gt = Variable(torch.from_numpy(im_gt.copy()),requires_grad=False).type(torch.cuda.FloatTensor).cuda()
        im_gt = im_gt/(torch.max(im_gt)-torch.min(im_gt))
        s = im_gt.shape
        GT = im_gt.view(s[0],s[1]*s[2])
        im_m = torch.matmul(p,GT).view(3,s[1],s[2])
        
        #Known the spatial downsampler
        #im_h = down_spa(torch.unsqueeze(im_gt,0))
        #im_h = Variable(torch.squeeze(im_h),requires_grad=False)
        

        #Unknown downsampler of spatial 
        im_h = data['H']
        im_h = Variable(torch.from_numpy(im_h), requires_grad=False).cuda()
    
        start_time = time.time()
        
        net_input = Variable(torch.unsqueeze(torch.rand_like(im_gt),0)).cuda()
        '''
        print('Stage one : pretrain the Spec_UpNet.\n')
        Spc_up = Spc_UpNet(im_h)
        H_RGB = Spc_up(torch.unsqueeze(im_m,0))
        print('Stage two : pretrain the Spa_UpNet.\n')
        Spa_up = Spa_UpNet(torch.squeeze(H_RGB,0))
        H_HSI = Spa_up(torch.unsqueeze(im_h,0))

        net_input =  Variable(0.8*H_RGB + 0.2*H_HSI).cuda()
        '''
        

        '''
        #learnable spectral downsampler
        class Down_spc(nn.Module):
            def __init__(self,in_channel=31, out_channel=3):
                super(Down_spc, self).__init__()
                self.conv2d_1 = nn.Conv2d(in_channels=in_channel, out_channels=31, kernel_size=1, padding=0)
                self.conv2d_2 = nn.Conv2d(in_channels=31, out_channels=out_channel, kernel_size=1, padding=0)
                self.conv2d_3 = nn.Conv2d(in_channels=31, out_channels=31, kernel_size=1, padding=0)

            def forward(self,x):
                out = self.conv2d_1(x)
                #out = self.conv2d_3(out)
                out = self.conv2d_2(out)
                return out

        down_spc = Down_spc(31,3).type(torch.cuda.FloatTensor).cuda()
        down_spc = PreT_DspcNet(down_spc,im_m,im_h)
        optimizer_spc = torch.optim.Adam(down_spc.parameters(),lr=lr,weight_decay=5e-4)
        '''

        #Learnable spatial downsampler
        KS = 32
        dow = nn.Sequential(nn.ReplicationPad2d(int((KS - factor)/2.)), nn.Conv2d(1,1,KS,factor))
        class Apply(nn.Module):
            def __init__(self, what, dim, *args):
                super(Apply, self).__init__()
                self.dim = dim
                self.what = what

            def forward(self, input):
                inputs = []
                for i in range(input.size(self.dim)):
                    inputs.append(self.what(input.narrow(self.dim, i, 1)))
                return torch.cat(inputs, dim=self.dim)

            def __len__(self):
                return len(self._modules)

        downs = Apply(dow, 1)
        downs = downs.cuda()
        
        optimizer_d = torch.optim.Adam(downs.parameters(),lr=lr,weight_decay=5e-4)
       


        f = open(save_path+name+'_result.txt','a+')
        f.write('\n\n\nThe result of PSNR & SAM is :\n\n')
        f.write('The experiment:At 8 times SR & the DSpa is {}.\n\n'.format(Dsets_name))
        f2 = open(save_path+name+'_loss.txt','w')
        f2.write('\n\n\nThe experiment loss is :\n\n')
        #start_time = time.time()
        #iteration
        psnr_max = 0
        sam_max = 0
        print('Stage three : Producing with the {} image'.format(name))
        for i in range(num_steps):
            #input data
            output = net(net_input)
            #procese of output
            S = output.shape
            #Dspa_O = down_spa(output)
            #Dspa_O = Dspa_O.view(Dspa_O.shape[1],Dspa_O.shape[2],Dspa_O.shape[3])
            Dspa_O = downs(output)
            Dspa_O = torch.squeeze(Dspa_O,0)
            out = output.view(S[1],S[2]*S[3])
            Dspc_O = torch.matmul(p,out).view(im_m.shape[0],S[2],S[3])
            #Dspc_O = torch.squeeze(down_spc(output),0)

            #zero the grad
            optimizer.zero_grad()
            #optimizer_spc.zero_grad()
            optimizer_d.zero_grad()


            loss = mse(Dspa_O,im_h)+mse(Dspc_O,im_m)
            #loss = L1Loss(Dspa_O,im_h) + L1Loss(Dspc_O,im_m)
       
            #backward the loss
            loss.backward()
            #optimize the parameter
            optimizer.step()
            #optimizer_spc.step()
            optimizer_d.step()

            #print('At step {},the loss is {}.'.format(i,loss.data.cpu()))

            if i%10 == 0:
                f2.write('At step {},the loss is {}\n'.format(i,loss.data.cpu()))
        
            if i%100 == 0: 
                output = Variable(output,requires_grad=False).cuda()
                output = output.view(S[1],S[2],S[3])
                psnr = PSNR_GPU(im_gt,output)
                sam = SAM_GPU(im_gt,output)
                f.write('{},{}\n'.format(psnr,sam))
                print('**{}**{}**At the {}th loop the PSNR&SAM is {},{}.'.format(k,Dsets_name,i,psnr,sam))
                if i%1000 == 0:
                    #change the learning rate
                    lr = 0.7*lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr 
                    #for param_group in optimizer_d.param_groups:
                    #    param_group['lr'] = lr
                if i == 5000:
                   data = {}
                   output = np.array(output)
                   data['data'] = output
                   sio.savemat(save_path+name+'_r.mat',data)

                if psnr > psnr_max:
                    data = {}
                    output = np.array(output)
                    data['data'] = output
                    sio.savemat(save_path+name+'_rM.mat',data)
                    psnr_max,sam_max = psnr,sam 

        #Out = Variable(torch.squeeze(output),requires_grad=False)
        #Out = np.array(out)
        #D = {}
        #D['data'] = output
        #sio.savemat(save_path+name+'_r.mat',D)
        print('The PSNR&SAM of {} is :{} , {}.'.format(name,psnr_max,sam_max))
        used_time = time.time()-start_time
        f.write('The training time is :{}.'.format(used_time))
        #torch.save(downs,'D_spa.pkl')
        #torch.save(down_spc,'D_spc.pkl')
        f.close()
        f2.close()

        torch.cuda.empty_cache()




