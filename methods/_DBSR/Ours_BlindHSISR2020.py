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
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import os
import torch.nn.functional as func

from DownKernel import *

##parameter setting
U_spa = 1                           # 0 indicates the spatial degeneration model is known, 1 is unknown
U_spc = 1                           # 0 indicates the spatial degeneration model is known, 1 is unknown
pretrain = 1                        # image specific prior learning
num_steps = 5001                    # iterations
lr_dc = 1.1e-4                      # learning rate of spectral degeneration network
lr_da = 1.1e-4                      # learning rate of spatial degeneration network
lr_i = 1.2e-3                       # learning rate of the generator network

factor = 8
kers = 1                            # the order of the spatial blur kernel in test phase
N_h = 40                            # SNR of the observed LR HSI
N_m = 40                            # SNR of the observed HR MSI
scale = 0                           # the noise intensity in test spectral response matrix
Dsets_name = 'Both Unknown X{} K{} HSI_{}N MSI_{}N P{}N'.format(factor, kers, N_h, N_m, scale)

CAVE_dataset = False

names = names_ICLV_test
dir_data = ''    # The path of ground truth HSI.
H_path = ''      # The path of the LR HSI .
M_path = ''      # The path of the HR MSI.
save_path = ''   # The path of where to save results.

if not os.path.exists(save_path):
    os.mkdir(save_path)



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

# The dataset of spatial downsampled HR MSI to HR MSI
class RGB_sets(data.Dataset):
    def __init__(self,RGB,Label,P_N,P_S,factor):
        self.RGB = RGB
        self.size = RGB.shape
        self.P_S = P_S
        self.Label = Label
        self.P_N = P_N
        self.D_spa = Downsampler(n_planes=31,factor=factor,kernel_type='gauss12',phase=0,preserve_size=True).type(torch.cuda.FloatTensor)

    def __getitem__(self, index):
        [p,q] = self.Label[index,:]
        P_S = self.P_S
        RGB = self.RGB[:,p:p+P_S,q:q+P_S]
        D_RGB = self.D_spa(torch.unsqueeze(RGB,0))

        return torch.squeeze(D_RGB,0),RGB

    def __len__(self):
        return self.P_N

# The dataset of spectral downsampled LR HSI to LR HSI
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


# Crop the test image into patches.
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


# Spatial upsample model
def Spa_UpNet(image):
    if factor == 8:
        P_S = 32
    elif factor == 16:
        P_S = 64
    else:
        P_S = 128
    Label, P_n = Get_Label(image.shape[1:], P_S)
    dataset = RGB_sets(image, Label, P_n, P_S,factor=factor)
    data = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    N1 = nn.Sequential()
    N1.add(get_net_1(31,'skip','reflection',n_channels=31,skip_n33d=40,skip_n33u=40,skip_n11=1,num_scales=2,upsample_mode='bilinear'))
    N1.add(nn.Upsample(scale_factor=factor,mode='bilinear'))
    N1.add(get_net_1(31,'skip','reflection',n_channels=31,skip_n33d=40,skip_n33u=40,skip_n11=1,num_scales=2,upsample_mode='bilinear'))
    N1 = N1.cuda()
    L1Loss = nn.L1Loss(reduce=True)
    optimizer_N1 = torch.optim.Adam(N1.parameters(), lr=5e-4)


    for epoch in range(500):
        running_loss = 0
        for i,batch in enumerate(data,1):
            lr,hr = batch[0],batch[1]
            lr,hr = Variable(lr).cuda(),Variable(hr).cuda()

            out = N1(lr)
            loss = L1Loss(out,hr)
            running_loss += loss.detach()
            optimizer_N1.zero_grad()
            loss.backward()
            optimizer_N1.step()
    return N1

# spectral upsample model
def Spc_UpNet(image):
    P_S = 16
    Label,P_n = Get_Label(image.shape[1:],P_S)
    dataset = HSI_sets(image,Label,P_n,P_S)
    data = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    N2 = get_net_1(3,'skip','reflection',n_channels=31,skip_n33d=31,skip_n33u=31,skip_n11=4,num_scales=5,upsample_mode='bilinear')
    N2 = N2.cuda()
    L1Loss = nn.L1Loss(reduce=True)
    optimizer_N2 = torch.optim.Adam(N2.parameters(), lr=5e-4)
    
    for epoch in range(500):
        running_loss = 0
        for i,batch in enumerate(data,1):
            lr,hr = batch[0],batch[1]
            lr,hr = Variable(lr).cuda(),Variable(hr).cuda()

            out = N2(lr)
            loss = L1Loss(out,hr)
            running_loss += loss.detach()
            optimizer_N2.zero_grad()
            loss.backward()
            optimizer_N2.step()
    return N2

# Learnable spectral response matrix
class L_Dspec(nn.Module):
    def __init__(self,in_channel,out_channel,P_init):
        super(L_Dspec, self).__init__()
        self.in_channle = in_channel
        self.out_channel = out_channel
        self.P = Parameter(P_init)

    def forward(self,input):
        S = input.shape
        out = torch.reshape(input,[S[1],S[2]*S[3]])
        out = torch.matmul(self.P,out)

        return torch.reshape(out,[self.out_channel,S[2],S[3]])


if __name__ =='__main__':

    # define the lossfunction
    L1Loss = nn.L1Loss()
    k = 0
    files = os.listdir(dir_data)

    for name in names:
        lr = lr_i
        k += 1

        # Load the Ground Truth HSI
        if CAVE_dataset:
            data = sio.loadmat(dir_data+name+'_ms.mat')
        else:
            data = sio.loadmat(dir_data+name+'.mat')
        print('Producing with the {} image'.format(name))
        im_gt = data['data']

        # Generate the spectral response matrix
        p = create_P()
        p = Variable(torch.from_numpy(p.copy()), requires_grad=False).cuda()

        # Trans the GT into Variable and nomalize it
        im_gt = Variable(torch.from_numpy(im_gt.copy()),requires_grad=False).type(torch.cuda.FloatTensor).cuda()
        im_gt = im_gt/(torch.max(im_gt)-torch.min(im_gt))
        s = im_gt.shape
        GT = im_gt.view(s[0],s[1]*s[2])

        # Load the observed HR MSI
        M_data = sio.loadmat(M_path+name+'.mat')
        im_m = M_data['HR_MSI']
        im_m = Variable(torch.from_numpy(im_m), requires_grad=False).type(torch.cuda.FloatTensor)

        if U_spc == 1:
            P_N = sio.loadmat('P_N.mat')
            P_N = torch.from_numpy(P_N['P'])
            down_spc = L_Dspec(31,3,P_N).type(torch.cuda.FloatTensor).cuda()
            optimizer_spc = torch.optim.Adam(down_spc.parameters(),lr=lr_dc,weight_decay=1e-5)

        # Load the Observed LR HSI
        H_data = sio.loadmat(H_path+name+'.mat')
        im_h = H_data['LR_HSI']
        im_h = Variable(torch.from_numpy(im_h), requires_grad=False).cuda()

        start_time = time.time()

        if pretrain == 1:
            print('Stage one : Pretrain the Spc_UpNet.')
            Spc_up = Spc_UpNet(im_h)
            H_RGB = Spc_up(torch.unsqueeze(im_m,0))
            print('Stage two : Pretrain the Spa_UpNet.')
            Spa_up = Spa_UpNet(torch.squeeze(H_RGB,0))
            H_HSI = Spa_up(torch.unsqueeze(im_h,0))
            net_input =  Variable(0.8*H_RGB + 0.2*H_HSI).cuda()
        else:
            net_input = Variable(torch.unsqueeze(torch.rand_like(im_gt), 0)).cuda()

        
        if U_spa == 1:

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
            optimizer_d = torch.optim.Adam(downs.parameters(),lr=lr_da,weight_decay=1e-5)
        else:
            down_spa = Downsampler(n_planes=im_gt.shape[0], factor=factor, kernel_type='gauss12', phase=0,preserve_size=True).type(torch.cuda.FloatTensor)

        # get_input
        net = get_net(im_gt.shape[0], 'skip', 'reflection', n_channels=im_gt.shape[0], skip_n33d=256, skip_n33u=256,skip_n11=1, num_scales=2, upsample_mode='bilinear')
        net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        f = open(save_path+name+'_result.txt','a+')
        f.write('\n\n\nThe result of PSNR & SAM is :\n\n')
        f.write('The experiment:At {} times SR & All the downsampler is unknown & the DSpa is {}.\n\n'.format(factor,Dsets_name))
        f2 = open(save_path+name+'_loss.txt','w')
        f2.write('\n\n\nThe experiment loss is :\n\n')
        psnr_max = 0
        sam_max = 0
        print('Stage three : Producing with the {} image'.format(name))
        for i in range(num_steps):

            #input data
            output = net(net_input)

            #procese of output
            S = output.shape
            if U_spa == 0:
                Dspa_O = down_spa(output)
                Dspa_O = Dspa_O.view(Dspa_O.shape[1],Dspa_O.shape[2],Dspa_O.shape[3])
            else:
                Dspa_O = downs(output)
                Dspa_O = torch.squeeze(Dspa_O,0)
            if U_spc == 0:
                out = output.view(S[1],S[2]*S[3])
                Dspc_O = torch.matmul(p,out).view(im_m.shape[0],S[2],S[3])
            else:
                Dspc_O = down_spc(output)

            #zero the grad
            optimizer.zero_grad()
            if U_spc==1:
                optimizer_spc.zero_grad()
            if U_spa==1:
                optimizer_d.zero_grad()

            loss = L1Loss(Dspa_O,im_h) + L1Loss(Dspc_O,im_m)
 
            #backward the loss
            loss.backward()

            #optimize the parameter
            optimizer.step()
            if U_spc==1:
                optimizer_spc.step()
            if U_spa==1:
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
                    lr = 0.7*lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr 
                    
            if i == num_steps-1:
               data = {}
               out = np.array(output.squeeze().detach().cpu())
               data['data'] = out
               sio.savemat(save_path+name+'_r.mat',data)
               if U_spa == 1:
                   torch.save(downs,save_path+name+'_DA.pth')
               if U_spc == 1:
                   torch.save(down_spc,save_path+name+'_DC.pth')

        used_time = time.time()-start_time
        print('The training time is :{}.'.format(used_time))
        f.write('The training time is :{}.'.format(used_time))
        f.close()
        f2.close()

        torch.cuda.empty_cache()




