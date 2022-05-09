import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

#There have five kind of kernels

class DownKernel(nn.Module):
    def __init__(self,K_num,factor,channels,preserve_size=True):
        super(DownKernel,self).__init__()

        Kernels = sio.loadmat('./f_set.mat')
        Kernels = Kernels['f_set']

        if K_num == 1:
            self.kernel = torch.from_numpy(Kernels[0,0])
        elif K_num == 2:
            self.kernel = torch.from_numpy(Kernels[0, 1])
        elif K_num == 3:
            self.kernel = torch.from_numpy(Kernels[0, 2])
        elif K_num == 4:
            self.kernel = torch.from_numpy(Kernels[0, 3])
        else:
            print('Out of the range of kernels number.')
            exit()

        self.K_S = self.kernel.shape
        pad = int((self.K_S[0]-1)/2.)
        downsampler = nn.Conv2d(channels, channels, kernel_size=self.K_S, stride=factor,padding=pad)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0
 
        #self.kernel = self.kernel.type(torch.DoubleTensor)
        #print(type(self.kernel))
        #print(self.kernel)
        for i in range(channels):
            downsampler.weight.data[i,i] = self.kernel
        #downsampler.weight.data.type(torch.DoubleTensor)
        self.downsampler_ = downsampler

        '''if preserve_size:
            if self.K_S[0] % 2 == 1:
                pad = int((self.K_S[0]-1)/2.)
            else:
                pad = int((self.K_S[0]-factor)/2.)

            self.padding = nn.ReplicationPad2d([pad,pad])

        self.preserve_size = preserve_size'''

    def forward(self,x):
        '''if self.preserve_size:
            self.x = self.padding(input)
        else:
            self.x = input'''
        self.x = x
        return self.downsampler_(self.x)




