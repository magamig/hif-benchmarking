import torch
import torch.nn as nn
# from SpaNet import BCR, denselayer
import numpy as np
import torch.nn.functional as f
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class My_Bn(nn.Module):
    def __init__(self):
        super(My_Bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)
        

class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False):
        super(BCR,self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                # self.Bn = nn.BatchNorm2d(num_features=cin)
                self.Bn = My_Bn()
                self.Module = nn.Sequential(
                    self.Bn,
                    self.conv,
                    self.relu
                )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                )
        else:
            if BN:
                # self.Bn = nn.BatchNorm2d(num_features=cin)
                self.Bn = My_Bn()
                self.Module = nn.Sequential(
                    self.Bn,
                    self.conv,
                )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class denselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=True,BN=True)
        self.actlayer = BCR(kernel=3,cin=cout,cout=cout,group=1,RELU=RELU,padding=1,BN=True)
        # self.Conv2d = BCR(kernel=3,cin=cin,cout=cout,group=1,RELU=RELU,padding=1,BN=False)
        self.bn = My_Bn()
    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)
        # output = self.bn(output) 
        return output

class Updenselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True):
        super(Updenselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=True,BN=False)
        self.actlayer = BCR(kernel=4,cin=cout,cout=cout,group=1,RELU=RELU,padding=1,stride = -2, BN=False)
        # self.Conv2d = BCR(kernel=3,cin=cin,cout=cout,group=1,RELU=RELU,padding=1,BN=False)
    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)
        return output
class recon_net(nn.Module):
    def __init__(self,cin,cout =31,RELU=True):
        super(recon_net,self).__init__()
        self.convlayer = nn.Sequential(
            Updenselayer(cin = 31,cout=62,RELU=True),
            denselayer(cin = 62,cout= 62),
            Updenselayer(cin = 62,cout=128,RELU=True),
            denselayer(cin = 128,cout= 128),
            Updenselayer(cin = 128,cout=31,RELU=False),
        )
        self.bn = My_Bn()
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x_ = self.pool(x)
        x = self.convlayer(x)
        x = self.bn(x)
        output = x+x_
        return output

class scale_learning(nn.Module):
    def __init__(self):
        super(scale_learning,self).__init__()
        self.dmodule = nn.Sequential(
            BCR(kernel = 4,cin = 31, cout = 64, group= 1, stride= 2,padding=1,BN=False),
            BCR(kernel = 4,cin = 64, cout = 64, group= 1, stride= 2,padding=1,BN=False),
            BCR(kernel = 4,cin = 64, cout = 64, group= 1, stride= 2,padding=1,BN=False),
            BCR(kernel = 4,cin = 64, cout = 31, group= 1, stride= 2,padding=1,BN=False,RELU=False))
        self.pool_layer = nn.AdaptiveAvgPool2d(output_size=1)
    def forward(self,x):
        feature_1 = self.pool_layer(x)
        feature_2 = self.dmodule(x)
        # feature_2 = torch.nn.functional.sigmoid(feature_2)
        feature_2 = feature_1+ self.pool_layer(feature_2)
        return feature_2

class stage(nn.Module):
    def __init__(self,cin,cout,final=False,extra=0):
        super(stage,self).__init__()
        self.Upconv = BCR(kernel = 3,cin = cin, cout = cout,stride= 1,padding=1)
        if final == True:
            f_cout = cout +1
        else:
            f_cout = cout
        mid = cout*3
        self.denselayers = nn.ModuleList([
            denselayer(cin=2*cout+extra,cout=cout*2),
            denselayer(cin=4*cout+extra,cout=cout*2),
            denselayer(cin=6*cout+extra,cout=cout*2),
            denselayer(cin=8*cout+extra,cout=cout*2),
            denselayer(cin=10*cout+extra,cout=cout*2),
            denselayer(cin=12*cout+extra,cout=cout*2),
            denselayer(cin=14*cout+extra,cout=cout*2),
            denselayer(cin=16*cout+extra,cout=f_cout,RELU=False)])
    def forward(self,HSI,MSI,extra_data=None):
        MSI = self.Upconv(MSI)
        if extra_data is not None:
            assert(MSI.shape == extra_data.shape)
        assert(MSI.shape == HSI.shape)
        if extra_data != None:
            x = torch.cat([HSI,MSI,extra_data],1)
        else:
            x = torch.cat([HSI,MSI],1)
        x = [x]
        for layer in self.denselayers:
            x_ = layer(torch.cat(x,1))
            x.append(x_)
        
        if extra_data is not None:
            if x[-1].shape[1] != HSI.shape[1]:
                output = torch.tanh(x[:,:HSI.shape[1],:,:]) + HSI
                output = torch.cat((output,torch.sigmoid(output[:,HSI.shape[1]:,:,:])),1)
            else:
                output = torch.tanh(x[-1]) + extra_data
        else:
            if x[-1].shape[1] != MSI.shape[1]:
                output = torch.tanh(x[-1][:,:MSI.shape[1],:,:]) + MSI
                output = torch.cat((output,torch.sigmoid(x[-1][:,MSI.shape[1]:,:,:])),1)
            else:
                output = torch.tanh(x[-1]) + MSI
        return output

class SpeNet(nn.Module):
    def __init__(self,extra=[0,0,0]):
        super(SpeNet,self).__init__()
        # self.stages = nn.ModuleList([
        #     stage(cin=3,cout=8,extra = extra[0]),
        #     stage(cin=8+3,cout=16,extra = extra[1]),
        #     stage(cin=16+3,cout=31,extra = extra[2],final=True)])
        self.stages = nn.ModuleList([
            stage(cin=3,cout=8,extra = extra[0]),
            stage(cin=8+3,cout=16,extra = extra[1]),
            stage(cin=16+3,cout=31,extra = extra[2],final=True)])
        self.scale_learning = scale_learning()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.refine = BCR(kernel = 3,cin = 31,cout = 31,stride=1,padding=1,BN=False)
        self.refine = nn.Sequential(
            denselayer(cin=31,cout=31))
        # self.recon_net = recon_net(cin = 31)
    def forward(self,HSI,MSI,extra_data=None):
        MSI = MSI - self.pool(MSI)
        MSI = [MSI]
        # scale = self.scale_learning(HSI)
        # HSI = HSI_.detach()- self.pool(HSI)
        # HSI = HSI- self.pool(HSI)
        ref = [np.array(range(8))*4, np.array(range(16))*2]
        ref[0][-1] = 30
        ref[1][-1] = 30
        for index , stage in enumerate(self.stages):
            if index <2:
                HSI_ = HSI[:,ref[index],:,:]
                if extra_data != None:
                    ex_data = extra_data[:,ref[index],:,:]
            else:
                HSI_ = HSI
                if extra_data != None:
                    ex_data = extra_data
            if index in [1,2]:
                if extra_data != None:
                    msi_ = stage(HSI_,torch.cat((MSI[index],MSI[0]),1),extra_data = ex_data)
                else:
                    msi_ = stage(HSI_,torch.cat((MSI[index],MSI[0]),1))
            else:
                if extra_data != None:
                    msi_ = stage(HSI_,MSI[index],extra_data = ex_data)
                else:
                    msi_ = stage(HSI_,MSI[index])

            MSI.append(msi_)
        refined_hsi = MSI[-1][:,:31,:,:]+HSI
        refined_hsi = self.refine(refined_hsi)+refined_hsi
        return MSI,HSI,refined_hsi



        
