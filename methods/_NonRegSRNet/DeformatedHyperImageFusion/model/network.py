"""
@Author: zhengke
@Date: 2020-07-19 21:16:56
@LastEditors: zhengke
@LastEditTime: 2020-07-20 06:45:39
@Description: main defination for the networks
@FilePath: \DeformatedHyperImageFusion\model\network.py
"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .spectralnorm import SpectralNorm
import numpy as np
import math
from torch.distributions import biject_to


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=opt.lr_decay_gamma, patience=opt.lr_decay_patience,
        )
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "mean_space":
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == "mean_channel":
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer("one", torch.tensor(1, dtype=torch.float))

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
        self.register_buffer("zero", torch.tensor(0.01, dtype=torch.float))

    def __call__(self, input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


def cross_correlation_loss(I, J, n):
    #     I = I.permute(0, 3, 1, 2)
    #     J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.cuda()
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1, 1))
    J_sum = torch.conv2d(J, sum_filter, padding=1, stride=(1, 1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1, 1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1, 1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1, 1))
    win_size = n ** 2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + np.finfo(float).eps)

    return torch.mean(cc)


def smooothing_loss(y_pred):
    #     dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    #     dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def vox_morph_loss(y, ytrue, n=9, lamda=0.01):
    cc = cross_correlation_loss(y, ytrue, n)
    sm = smooothing_loss(y)
    # print(cc)
    # print(sm)
    # print("CC Loss", cc, "Gradient Loss", sm)
    loss = -1.0 * cc + lamda * sm
    return loss


class VoxMorphLoss(nn.Module):
    def __init__(self, n=9, lamda=0.1):
        super(VoxMorphLoss, self).__init__()
        self.n = n
        self.lamda = lamda

    def __call__(self, y, ytrue):
        loss = vox_morph_loss(y, ytrue, self.n, self.lamda)
        return loss


def gradient_loss(s, penalty="l2"):
    dy = torch.abs(s[:, 1:, :, :] - s[:, :-1, :, :])
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dz = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if penalty == "l2":
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2

    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, "conv%dd" % ndims)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = 1
        padding = pad_no
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


class ResBlock(nn.Module):
    def __init__(self, input_ch):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_ch, input_ch, 1, 1, 0), nn.LeakyReLU(0.2, True), nn.Conv2d(input_ch, input_ch, 1, 1, 0),
        )

    def forward(self, x):
        out = self.net(x)
        return out + x


def define_displacementfiled(in_channels, out_channels, gpu_ids, init_type="kaiming", init_gain=0.02):
    net = UNet(in_channels, out_channels)
    # import ipdb
    # ipdb.set_trace()
    # net = DisplacementField(in_channels, out_channels)
    return init_net(net, init_type, init_gain, gpu_ids)


class ResBlock3x3(nn.Module):
    def __init__(self, input_feas, feas):
        super(ResBlock3x3, self).__init__()
        self.net = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(input_feas, feas, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(feas, input_feas, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class DisplacementField(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DisplacementField, self).__init__()
        feas = 128
        self.first_conv = nn.Conv2d(in_channels, feas, 3, 1, 1)
        self.encoder_1 = ResBlock3x3(feas, feas)
        self.encoder_1_down = nn.Conv2d(feas, feas, 3, 2, 1)
        self.encoder_2 = ResBlock3x3(feas, feas)
        self.encoder_2_down = nn.Conv2d(feas, feas, 3, 2, 1)
        self.encoder_3 = ResBlock3x3(feas, feas)
        self.encoder_3_down = nn.Conv2d(feas, feas, 3, 2, 1)
        self.encoder_4 = ResBlock3x3(feas, feas)
        self.decoder_3_up = nn.ConvTranspose2d(feas, feas, 3, 2, 1, 1)
        self.decoder_3 = ResBlock3x3(feas, feas)
        self.decoder_2_up = nn.ConvTranspose2d(feas, feas, 3, 2, 1, 1)
        self.decoder_2 = ResBlock3x3(feas, feas)
        self.deocder_1_up = nn.ConvTranspose2d(feas, feas, 3, 2, 1, 1)
        self.decoder_1 = ResBlock3x3(feas, feas)
        self.final_conv = nn.Conv2d(feas, out_channels, 3, 1, 1)

    def forward(self, x):
        firs = self.first_conv(x)
        en_1 = self.encoder_1(firs)
        en_1_down = self.encoder_1_down(en_1)
        en_2 = self.encoder_2(en_1_down)
        en_2_down = self.encoder_2_down(en_2)
        en_3 = self.encoder_3(en_2_down)
        en_3_down = self.encoder_3_down(en_3)
        en_4 = self.encoder_4(en_3_down)
        de_3_up = self.decoder_3_up(en_4) + en_3
        de_3 = self.decoder_3(de_3_up)
        de_2_up = self.decoder_2_up(de_3) + en_2
        de_2 = self.decoder_2(de_2_up)
        de_1_up = self.deocder_1_up(de_2) + en_1
        # import ipdb
        # ipdb.set_trace()
        de_1 = self.decoder_1(de_1_up)
        out = self.final_conv(de_1)
        return out


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            # torch.nn.ConvTranspose2d(
            #     in_channels=mid_channel,
            #     out_channels=out_channels,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     output_padding=1,
            # ),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=1)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        # self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1,),
            torch.nn.BatchNorm2d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            # torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            # torch.nn.BatchNorm2d(mid_channel),
            # torch.nn.ReLU(),
        )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, resize=True):
        """
        This layer resize the layer from contraction block and concat it with expansive block vector
        """
        if resize:
            # c = (bypass.size()[2] - upsampled.size()[2]) // 2
            # bypass = F.pad(bypass, (-c, -c, -c, -c))
            bypass = nn.functional.interpolate(bypass, upsampled.size()[2:], mode="bilinear", align_corners=False)
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        

        # Encode
        encode_block1 = self.conv_encode1(x)
        # encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_block1)
        # encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_block2)
        # encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_block3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        # decode_block1 = nn.functional.interpolate(decode_block1, x.size()[2:], mode="bilinear", align_corners=False)
        final_layer = self.final_layer(decode_block1)

        # import ipdb
        # ipdb.set_trace()
        return final_layer


def define_spatial_transform(img_size, use_gpu=True):
    net = SpatialTransformation(use_gpu)
    # net = SpatialTransformer(img_size)
    return net


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode="bilinear"):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, mode=self.mode)


class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(
            torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0),
        )
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]),)

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu == True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):

        im = F.pad(im, (0, 0, 1, 1, 1, 1, 0, 0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = self.repeat(torch.arange(0, batch_size) * dim1, out_height * out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1, 0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1, 0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1, 0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1, 0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1, 0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1, 0)
        wb = (dx * (1 - dy)).transpose(1, 0)
        wc = ((1 - dx) * dy).transpose(1, 0)
        wd = ((1 - dx) * (1 - dy)).transpose(1, 0)

        output = torch.sum(torch.squeeze(torch.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])

        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


def define_msi2Abundance(input_ch, output_ch, gpu_ids, init_type="kaiming", init_gain=0.02, useClamp=True):

    net = Msi2A(input_c=input_ch, output_c=output_ch, ngf=64, useClamp=useClamp)

    return init_net(net, init_type, init_gain, gpu_ids)


class Msi2A(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, useClamp=True):
        super(Msi2A, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, output_c, 1, 1, 0),
        )

        self.softmax = nn.Softmax(dim=1)
        self.useClamp = useClamp

    def forward(self, x):
        if self.useClamp:
            return self.net(x).clamp_(0, 1)
        else:
            return self.softmax(self.net(x))


def define_A2img(input_ch, output_ch, gpu_ids, init_type="kaiming", init_gain=0.02):
    net = A2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)


class A2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(A2Img, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),)

    def forward(self, x):
        return self.net(x).clamp_(0, 1)

class A2ImgNONCLIP(nn.Module):
    def __init__(self, input_c, output_c):
        super(A2ImgNONCLIP, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, input_c * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_c * 2, input_c * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_c * 4, input_c * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_c * 2, output_c, 1, 1, 0),
            nn.ReLU(),
            )

    def forward(self, x):
        
        return self.net(x).clamp_(0, 1)
        


def define_HSI2A(input_ch, output_ch, gpu_ids, init_type="kaiming", init_gain=0.02, useClamp=True):

    net = HSI2A(input_c=input_ch, output_c=output_ch, ngf=64, useClamp=useClamp)
    return init_net(net, init_type, init_gain, gpu_ids)


class HSI2A(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, useClamp=True):
        super(HSI2A, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, output_c, 1, 1, 0),
            # nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=1)
        self.useClamp = useClamp

    def forward(self, x):
        if self.useClamp:
            return self.net(x).clamp_(0, 1)
        else:
            return self.softmax(self.net(x))


def define_psf(scale, gpu_ids, init_type="mean_space", init_gain=0.02):
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
        return torch.cat([self.net(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1,)


def define_hsi2msi(
    args, hsi_channels, msi_channels, sp_matrix, sp_range, gpu_ids, init_type="mean_channel", init_gain=0.02,
):
    if not args.isCalSP:
        net = MatrixDotHsi2Msi(sp_matrix)
    elif args.isCalSP:
        net = ConvolutionHsi2Msi(hsi_channels, msi_channels, sp_range)
        
        # net = MSIDecoder(hsi_channels, msi_channels, gpu_ids)
        # net = PredMuSigma(hsi_channels, msi_channels, gpu_ids)
        # net = LearnNormalDistribution(hsi_channels, msi_channels, gpu_ids, sp_range)
    return init_net(net, init_type, init_gain, gpu_ids)
    # return net


class SpectralAttention(nn.Module):
    def __init__(self, hsi_channels, msi_channels):
        super(SpectralAttention, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(hsi_channels, hsi_channels, 1, 1, 0))
        self.conv2 = nn.Sequential(nn.Conv2d(hsi_channels, hsi_channels, 1, 1, 0))

        self.linear = nn.Sequential(
            nn.Linear(hsi_channels * hsi_channels, msi_channels * msi_channels),
            nn.Linear(msi_channels * msi_channels, msi_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1(x).permute(1, 0, 2, 3).view(c, -1)
        x2 = self.conv2(x).permute(1, 0, 2, 3).view(c, -1).t()

        x1x2 = torch.mm(x1, x2).view(-1, c * c)
        x3 = self.linear(x1x2)

        return x3

class LearnNormalDistribution(nn.Module):
    def __init__(self, hsi_channels, msi_channels, gpu_ids, sp_range):
        super(LearnNormalDistribution, self).__init__()

        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels

        # sp_range = torch.mean(sp_range, dim=1)
        # import ipdb
        # ipdb.set_trace()

        self.dict_list = []
        for i in range(msi_channels):
            self.dict_list.append({
                "loc":torch.nn.parameter.Parameter(torch.tensor(sp_range.mean(axis=1)[i]+1, requires_grad=True)),
                "scale":torch.nn.parameter.Parameter(torch.tensor([1.0], requires_grad=True))})
            self.register_parameter(name="NormalDist_loc_%d"%(i), param=self.dict_list[i]["loc"])
            self.register_parameter(name="NormalDist_scale_%d"%(i), param=self.dict_list[i]["scale"])

        self.normal_dist = torch.distributions.Normal(0, 1)

        self.bands_range = torch.arange(1, self.hsi_channels+1).view(1, -1, 1, 1).to(torch.device("cuda:{}".format(gpu_ids[0])))

    def get_params(self):
        return self.dict_list

    def get_distribution(self, params):
        return torch.distributions.Normal(**self.get_constrained_params(params))

    def get_constrained_params(self, params):
        # constraints = self.normal_dist.arg_constraints
        # constrined_params = {}
        # for k, v in params.items():
        #     constrined_params[k] = biject_to(constraints[k])(v)
        # return constrined_params
        return params

    def forward(self, x):

        print(self.get_params())

        out_msi = []
        for i in range(self.msi_channels):
            dist = self.get_distribution(self.dict_list[i])
            prob = dist.log_prob(self.bands_range).exp()
            

            x_mul = x * prob
            sum_x_mul = torch.sum(x_mul, dim=1, keepdim=True)
            sum_prob = torch.sum(prob, dim=1, keepdim=True)
            out_msi.append(sum_x_mul / sum_prob)
        return torch.cat(out_msi, dim=1)


class PredMuSigma(nn.Module):
    def __init__(self, hsi_channels, msi_channels, gpu_ids):
        super(PredMuSigma, self).__init__()

        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels

        self.mu = SpectralAttention(hsi_channels, msi_channels)
        self.sigma = SpectralAttention(hsi_channels, msi_channels)

        self.band_vector = torch.arange(1, hsi_channels + 1).float().to(torch.device("cuda:{}".format(gpu_ids[0])))

    def forward(self, x):
        mu = self.mu(x).view(-1) * self.hsi_channels
        sigma = self.sigma(x).view(-1)

        print("mu:", mu.data.cpu())
        print("sigma", sigma.data.cpu())

        out_msi = []

        for i in range(self.msi_channels):
            gaussian_vector = self.gaussian(self.band_vector, mu[i], sigma[i])

            gaussian_vector = gaussian_vector.view(1, x.shape[1], 1, 1)
            vector_multiply_hsi = x * gaussian_vector.expand_as(x)
            sum_vector_multiply_hsi = torch.sum(vector_multiply_hsi, dim=1, keepdim=True)
            sum_gaussian_vector = torch.sum(gaussian_vector, dim=1, keepdim=True)
            out_msi.append(sum_vector_multiply_hsi / sum_gaussian_vector)
        return torch.cat(out_msi, dim=1)

    def gaussian(self, x, mu, sigma):

        y = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * math.pi))
        return y


class MSIDecoder(nn.Module):
    def __init__(self, hsi_channels, msi_channels, gpu_ids):
        super(MSIDecoder, self).__init__()

        self.msi_channels = msi_channels
        self.mu = torch.nn.Parameter(
            torch.ones(msi_channels, requires_grad=True).to(torch.device("cuda:{}".format(gpu_ids[0])))
        )
        
        self.sigma = torch.nn.Parameter(
            torch.ones(msi_channels, requires_grad=True).to(torch.device("cuda:{}".format(gpu_ids[0])))
        )
        self.band_vector = (
            torch.arange(1, hsi_channels + 1, requires_grad=False)
            .float()
            .to(torch.device("cuda:{}".format(gpu_ids[0])))
        )
        # if torch.cuda.is_available():
        #     self.mu = self.mu.cuda()
        #     self.sigma = self.sigma.cuda()
        #     self.band_vector = self.band_vector.cuda()
        # torch.nn.Parameter(self.sigma)
        # torch.nn.Parameter(self.mu)

        # self.register_parameter(self.sigma)
        # self.register_parameter(self.mu)

    def forward(self, x):

        print("mu:", self.mu.data.cpu())
        print("sigma", self.sigma.data.cpu())

        out_msi = []

        for i in range(self.msi_channels):
            gaussian_vector = self.gaussian(self.band_vector, self.mu[i], self.sigma[i])

            if torch.cuda.is_available():
                gaussian_vector = gaussian_vector.cuda()
            gaussian_vector = gaussian_vector.view(1, x.shape[1], 1, 1)
            vector_multiply_hsi = x * gaussian_vector.expand_as(x)
            sum_vector_multiply_hsi = torch.sum(vector_multiply_hsi, dim=1, keepdim=True)
            sum_gaussian_vector = torch.sum(gaussian_vector, dim=1, keepdim=True)
            out_msi.append(sum_vector_multiply_hsi / sum_gaussian_vector)
        return torch.cat(out_msi, dim=1)

    def gaussian(self, x, mu, sigma):

        y = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * math.pi))
        return y


class ConvolutionHsi2Msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(ConvolutionHsi2Msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:, 1] - self.sp_range[:, 0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        # import ipdb
        # ipdb.set_trace()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(x, 1, 1, 1, 0, bias=False) for x in self.length_of_each_band])
        # self.scale_factor_net = nn.Conv2d(1,1,1,1,0,bias=False)

    def forward(self, input):
        # batch,channel,height,weight = list(input.size())
        # scaled_intput = torch.cat([self.scale_factor_net(input[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:, self.sp_range[i, 0] : self.sp_range[i, 1] + 1, :, :]
            # input_slice = scaled_intput[:, :, :, :]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list, 1).clamp_(0, 1)


class MatrixDotHsi2Msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(MatrixDotHsi2Msi, self).__init__()
        self.register_buffer("sp_matrix", torch.tensor(spectral_response_matrix.transpose(1, 0)).float())

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(
            self.sp_matrix.expand(batch, -1, -1), torch.reshape(x, (batch, channel_hsi, heigth * width)),
        ).view(batch, channel_msi_sp, heigth, width)
        return hmsi


def define_d_msi(input_hmsi_ch, gpu_ids, init_type="kaiming", init_gain=0.02):
    net = DiscriminatorMSI(hmsi_c=input_hmsi_ch)
    return init_net(net, init_type, init_gain, gpu_ids)


class DiscriminatorMSI(nn.Module):
    def __init__(self, hmsi_c, ngf=64):
        super(DiscriminatorMSI, self).__init__()
        self.net_msi = [
            # nn.Conv2d(hmsi_c, ngf, kernel_size=3, stride=1, padding=1),
            SpectralNorm(nn.Conv2d(hmsi_c, ngf, kernel_size=3, stride=1, padding=1)),
            # nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
            SpectralNorm(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1)),
            # nn.InstanceNorm2d(ngf * 2),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1),
            # SpectralNorm(nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1)),
            # nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ngf * 2, 1, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
            SpectralNorm(nn.Conv2d(ngf * 2, 1, kernel_size=3, stride=1, padding=1)),
            nn.Sigmoid(),
        ]
        self.net_msi = nn.Sequential(*self.net_msi)

    def forward(self, hmsi):
        return self.net_msi(hmsi)


class NonZeroClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w.clamp_(0, 1e8)


class ZeroOneClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w.clamp_(0, 1)


class SumToOneClipper(object):
    def __call__(self, module):
        if hasattr(module, "weight"):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0, 10)
                w.div_(w.sum(dim=1, keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0, 5)

