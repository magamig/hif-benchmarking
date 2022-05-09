import torch
import torch.nn as nn
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

class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False):
        super(BCR,self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)))
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        if RELU:
            if BN:
                self.Bn = nn.BatchNorm2d(num_features=cin)
                # self.Bn = nn.InstanceNorm2d(num_features=cin,affine=False,track_running_stats=True)
                self.Module = nn.Sequential(
                    self.Bn,
                    self.conv,
                    self.relu
                    # self.Swish
                )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                    # self.Swish
                )
        else:
            if BN:
                self.Bn = nn.BatchNorm2d(num_features=cin)
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
    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)
        # output = self.Conv2d(x)
        return output

class stage(nn.Module):
    def __init__(self,upScale=4,final = False,extra = 0):
        super(stage,self).__init__()
        if upScale == 4:
            self.UpConv = nn.Sequential(
                BCR(kernel = 2, cin = 31, cout = 31, group=1,stride=-2),
                BCR(kernel = 2, cin = 31, cout = 31, group=1,stride=-2))
        elif upScale == 2:
            self.UpConv = nn.Sequential(BCR(kernel = 2, cin = 31, cout = 31, group=1,stride=-2))
        if final == True:
            f_out = 31 + 1
        else:
            f_out = 31
        self.DLayers = nn.ModuleList([
            denselayer(cin =34+extra,  cout=31),
            denselayer(cin =65+extra,  cout=31),
            denselayer(cin =96+extra,  cout=31),
            denselayer(cin =127+extra, cout=31),
            denselayer(158, f_out,RELU=False)])
    def forward(self,HSI,MSI,extra_data = None):
        # print(HSI.shape,'---------------------------------------------------------------------------------')
        HSI = self.UpConv(HSI)
        x = torch.cat([HSI,MSI],1)
        x = [x]
        for layer in self.DLayers:
            x_ = layer(torch.cat(x,1))
            x.append(x_)
        if extra_data is not None:
            add_data = extra_data
        else:
            add_data = HSI

        if extra_data is not None:

            output = torch.tanh(x[-1]) + add_data

        else:
            if HSI.shape[1] == x[-1].shape[1]:

                output = torch.tanh(x[-1]) + add_data

            else:

                output = torch.tanh(x[-1][:,:add_data.shape[1],:,:])+add_data
                output = torch.cat((output,torch.sigmoid(x[-1][:,add_data.shape[1]:,:,:])),1)

        return output

class SpaNet(nn.Module):
    #MSI input #
    def __init__(self,extra = 0 ):
        super(SpaNet,self).__init__()
        self.stages = nn.ModuleList([
            stage(upScale=2,extra= extra),
            stage(upScale=2,extra= extra),
            stage(upScale=2,extra= extra,final=True)])
    def forward(self,HSI,MSI,extra_data = [None,None,None]):
        assert(len(MSI)==3)
        HSI = [HSI]
        for index, stage in enumerate(self.stages):
            HSI_ = stage(HSI[index],MSI[index],extra_data= extra_data[index])
            # HSI_[:,:31,:,:] = torch.sigmoid(HSI_[:,:31,:,:])
            HSI.append(HSI_)
        return HSI