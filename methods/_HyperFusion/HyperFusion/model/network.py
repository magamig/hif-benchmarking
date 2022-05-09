import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))

        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        # print(input[0,:,:])
        loss = self.loss(input, target_tensor)
        # loss = torch.sum(torch.abs(target_tensor - input))
        return loss

def kl_divergence(p, q):

    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2

class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self,input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss

class ResBlock(nn.Module):
    def __init__(self, input_ch):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_ch, input_ch, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        )
    def forward(self, x):
        out = self.net(x)
        return out + x

def define_msi2s(input_ch, output_ch,gpu_ids, n_res, init_type='kaiming', init_gain=0.02, useSoftmax=True):

    net = Msi2Delta(input_c=input_ch, output_c=output_ch, ngf=64, n_res=n_res, useSoftmax=useSoftmax)

    return init_net(net, init_type, init_gain, gpu_ids)

class Msi2Delta(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, n_res=3, useSoftmax=True):
        super(Msi2Delta, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf*2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*2, ngf*4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*4, ngf*8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*8, output_c, 1, 1, 0),
            nn.ReLU()
            # nn.Tanh()
        )
        # self.net_in = nn.Conv2d(input_c, ngf*2, 1, 1, 0)
        # self.net_mid = nn.Sequential(*[ResBlock(ngf*2) for _ in range(n_res)])
        # self.net_out = nn.Conv2d(ngf*2, output_c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)
        self.usefostmax = useSoftmax

    def forward(self, x):
        if self.usefostmax == True:
            return self.softmax(self.net(x))
        elif self.usefostmax == False:
            return self.net(x)
        # return self.net_out(self.net_mid(self.net_in(x)))

def define_s2img(input_ch, output_ch,gpu_ids, init_type='kaiming', init_gain=0.02):
    net = S2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)

class S2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(S2Img, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

def define_lr2s(input_ch, output_ch,gpu_ids, n_res, init_type='kaiming', init_gain=0.02, useSoftmax=True):

    net = Lr2Delta(input_c=input_ch, output_c=output_ch, ngf=64, n_res=n_res, useSoftmax=useSoftmax)
    return init_net(net, init_type, init_gain, gpu_ids)

class Lr2Delta(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, n_res=3, useSoftmax=True):
        super(Lr2Delta, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf*2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*2, ngf*4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*4, ngf*8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*8, output_c, 1, 1, 0),
            nn.ReLU()
            
        )
        self.softmax = nn.Softmax(dim=1)
        self.net_in = nn.Conv2d(input_c, ngf*2, 1, 1, 0)
        self.net_mid = nn.Sequential(*[ResBlock(ngf*2) for _ in range(n_res)])
        self.net_out = nn.Conv2d(ngf*2, output_c, 1, 1, 0)
        self.usesoftmax = useSoftmax
    def forward(self, x):
        if self.usesoftmax == True:
            return self.softmax(self.net(x))
        elif self.usesoftmax == False:
            return self.net(x)
        # return self.net(x).clamp_(0,1)
        # return self.net_out(self.net_mid(self.net_in(x)))


def define_psf(scale, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = PSF(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)

class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)

def define_hr2msi(args, hsi_channels, msi_channels, sp_matrix, sp_range, gpu_ids, init_type='mean_channel', init_gain=0.02):
    if args.isCalSP == False:
        net = matrix_dot_hr2msi(sp_matrix)
    elif args.isCalSP == True:
        net = convolution_hr2msi(hsi_channels, msi_channels, sp_range)
    return init_net(net, init_type, init_gain, gpu_ids)

class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:,1] - self.sp_range[:,0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        # import ipdb
        # ipdb.set_trace()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(x,1,1,1,0,bias=False) for x in self.length_of_each_band])
        # self.scale_factor_net = nn.Conv2d(1,1,1,1,0,bias=False)

    def forward(self, input):
        # batch,channel,height,weight = list(input.size())
        # scaled_intput = torch.cat([self.scale_factor_net(input[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:,self.sp_range[i,0]:self.sp_range[i,1]+1,:,:]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list,1).clamp_(0,1)

class matrix_dot_hr2msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(matrix_dot_hr2msi, self).__init__()
        self.register_buffer('sp_matrix', torch.tensor(spectral_response_matrix.transpose(1,0)).float())

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(self.sp_matrix.expand(batch,-1,-1),
                         torch.reshape(x, (batch, channel_hsi, heigth*width))).view(batch,channel_msi_sp, heigth, width)
        return hmsi


class NormGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(NormGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class NonZeroClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1e8)

class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1)

class SumToOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0,10)
                w.div_(w.sum(dim=1,keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0,5)
