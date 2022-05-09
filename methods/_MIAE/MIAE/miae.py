# Model Inspired Autoencoder for Unsupervised Hyperspectral Image Superresolution, TGRS
# Author: JianJun Liu
# Date: 2022-1-13
import numpy as np
import scipy.io as sio
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
import torch.utils.data as data
from MIAE.utils import toolkits, torchkits, DataInfo, BlurDown, PatchDataset
from MIAE.blind import Blind

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AENet(nn.Module):
    def __init__(self, hs_bands, ms_bands, edm_num, stage=3):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.edm_num = edm_num
        self.stage = stage
        self.module_list = nn.ModuleList([])
        edm = torch.ones([self.hs_bands, self.edm_num, 1, 1]) * (1.0 / self.edm_num)
        self.edm = nn.Parameter(edm)
        self.Y_net = nn.Sequential(
            nn.Conv2d(self.hs_bands, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.Z_net = nn.Sequential(
            nn.Conv2d(self.ms_bands, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.S0_net = nn.Sequential(
            nn.Conv2d(2 * self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        for k in range(0, self.stage - 1):
            self.module_list.append(
                nn.Sequential(
                    nn.Conv2d(self.edm_num, self.edm_num, (1, 1)),
                    nn.LeakyReLU(0.2)
                )
            )
            self.module_list.append(
                nn.Sequential(
                    nn.Conv2d(3 * self.edm_num, self.edm_num, (1, 1)),
                    nn.LeakyReLU(0.2)
                )
            )
        self._init_weights(self)
        pass

    def forward(self, Yu, Z):
        N, B, H, W = Yu.shape
        N, b, H, W = Z.shape
        Y1 = self.Y_net(Yu)
        Z1 = self.Z_net(Z)
        S = torch.cat([Y1, Z1], dim=1)
        S = self.S0_net(S)
        for k in range(0, self.stage - 1):
            S = self.module_list[2 * k - 2](S)
            S = torch.cat([S, Y1, Z1], dim=1)
            S = self.module_list[2 * k - 1](S)
        S = torch.clamp(S, 0.0, 1.0)
        X = fun.conv2d(S, self.edm, None)
        X = torch.clamp(X, 0.0, 1.0)
        return X

    @staticmethod
    def _init_weights(model, init_type='normal'):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                num_inputs = m.weight.data.shape[1]
                if init_type == 'normal':
                    nn.init.trunc_normal_(m.weight.data, mean=0.0, std=np.sqrt(1.0 / num_inputs))
                elif init_type == 'constant':
                    nn.init.constant_(m.weight.data, 1.0 / num_inputs)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        pass


class MIAE(DataInfo):
    def __init__(self, ndata, nratio, nsnr=0, psf=None, srf=None, edm_num=80, stage=3):
        super().__init__(ndata, nratio, nsnr)
        self.strX = 'X.mat'
        if psf is not None:
            self.psf = psf
        if srf is not None:
            self.srf = srf
        # set
        self.lr = 0.005  # learning rate
        self.edm_num = edm_num  # (like) endmember number
        self.ker_size = self.psf.shape[0]  # spatial blur kernel size
        self.patch_size = 5 * self.ratio  # divide the image into patches to accelerate training
        self.patch_stride = 3 * self.ratio  # make sure the patches are overlapped so that all pixels are included
        self.batch_size = self.set_batch_size()  # depending on the spatial size of image
        self.lam_A, self.lam_B, self.lam_C = 1, 1, 1e-3  # weights for spectral term, spatial term and weight decay
        self.lr_fun = lambda epoch: (1.0 - max(0, epoch + 1 - 1000) / 9000)  # decay of learning rate
        # define
        self.psf = np.reshape(self.psf, newshape=(1, 1, self.ker_size, self.ker_size))
        self.psf = torch.tensor(self.psf).cuda()
        self.psf_hs = self.psf.repeat(self.hs_bands, 1, 1, 1)
        self.srf = np.reshape(self.srf, newshape=(self.ms_bands, self.hs_bands, 1, 1))
        self.srf = torch.tensor(self.srf).cuda()
        # variable, graph and etc
        self.__hsi = torch.tensor(self.hsi)
        self.__msi = torch.tensor(self.msi)
        self.__hsi_up = nn.Upsample(scale_factor=self.ratio, mode='bilinear', align_corners=False)(self.__hsi)
        self.model = AENet(self.hs_bands, self.ms_bands, self.edm_num, stage).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lam_C)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fun)
        toolkits.check_dir(self.model_save_path)
        torchkits.get_param_num(self.model)
        self.hs_border = math.ceil((self.ker_size - 1) / 2 / self.ratio)  # remove the pixels effected by spatial blur
        self.ms_border = self.hs_border * self.ratio  # remove the pixels effected by spatial blur
        self.dataset = PatchDataset(self.__hsi, self.__msi, self.__hsi_up,
                                    self.patch_size, self.patch_stride, self.ratio)
        self.blur_down = BlurDown()
        pass

    def set_batch_size(self):
        batch_size = 100
        if self.ratio == 4:
            batch_size = 100
        if self.ratio == 8:
            batch_size = 25
        if self.ratio == 16:
            batch_size = 5
        batch_size = (self.height // 512) * (self.width // 256) * batch_size
        return batch_size

    def cpt_target(self, X):
        Y = self.blur_down(X, self.psf_hs, int((self.ker_size - 1) / 2), self.hs_bands, self.ratio)
        Z = fun.conv2d(X, self.srf, None)
        return Y, Z

    def build_loss(self, Y, Z, hsi, msi):
        dY = Y - hsi
        dZ = Z - msi
        dY = dY[:, :, self.hs_border: -self.hs_border, self.hs_border: -self.hs_border]
        dZ = dZ[:, :, self.ms_border: -self.ms_border, self.ms_border: -self.ms_border]
        loss = self.lam_A * torch.sum(torch.mean(torch.abs(dY), dim=(2, 3))) * (self.height / self.ratio) * (
                self.width / self.ratio)
        loss += self.lam_B * torch.sum(torch.mean(torch.abs(dZ), dim=(2, 3))) * self.width * self.height
        return loss

    def train(self, max_iter=10000, verb=True, is_save=True):
        # train ...
        loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        iteration, epoch = 0, 0
        time_start = time.perf_counter()
        self.model.train()
        while True:
            # train
            for i, (hsi, msi, hsi_up, item) in enumerate(loader):
                hsi, msi, hsi_up = hsi.cuda(), msi.cuda(), hsi_up.cuda()
                X = self.model(hsi_up, msi)
                Yhat, Zhat = self.cpt_target(X)
                loss = self.build_loss(Yhat, Zhat, hsi, msi)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.edm.data.clamp_(0.0, 1.0)  # check decoder weight
                iteration += 1
                # evaluation
                if verb is True:
                    if iteration % 1000 == 0:
                        self.evaluation(iteration)
                        self.model.train()
                if iteration >= max_iter:
                    break
                self.scheduler.step()
                pass
            epoch += 1
            if iteration >= max_iter:
                if verb is True:
                    self.evaluation(epoch)
                break
            pass
        time_end = time.perf_counter()
        train_time = time_end - time_start
        print('running time %ss' % train_time)
        X, test_time = self.evaluation(epoch)
        if is_save is True:
            torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')
            sio.savemat(self.save_path + self.strX, {'X': X, 't1': train_time, 't2': test_time})
        pass

    def evaluation(self, iteration):
        self.model.eval()
        lr = self.optimizer.param_groups[0]['lr']
        t0 = time.perf_counter()
        X = self.model(self.__hsi_up.cuda(), self.__msi.cuda())
        test_time = time.perf_counter() - t0
        Yhat, Zhat = self.cpt_target(X)
        loss = self.build_loss(Yhat, Zhat, self.__hsi.cuda(), self.__msi.cuda())
        Xh = torchkits.to_numpy(X)
        Xh = toolkits.channel_last(Xh)
        if self.ref is not None:
            psnr = toolkits.compute_psnr(self.ref, Xh)
            sam = toolkits.compute_sam(self.ref, Xh)
            print('iter/epoch: %s, lr: %s, psnr: %s, sam: %s, loss: %s' % (iteration, lr, psnr, sam, loss))
        else:
            print('iter/epoch: %s, lr: %s, loss: %s' % (iteration, lr, loss))
        return Xh, test_time

    def train_all(self, max_iter=10000, verb=True, is_save=True):
        time_start = time.perf_counter()
        hsi, msi, hsi_up = self.__hsi.cuda(), self.__msi.cuda(), self.__hsi_up.cuda()
        self.model.train()
        for epoch in range(0, max_iter):
            X = self.model(hsi_up, msi)
            Yhat, Zhat = self.cpt_target(X)
            loss = self.build_loss(Yhat, Zhat, hsi, msi)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.edm.data.clamp_(0.0, 1.0)
            # test
            if verb is True:
                if (epoch + 1) % 1000 == 0:
                    self.evaluation(epoch + 1)
                    self.model.train()
            self.scheduler.step()
            pass
        time_end = time.perf_counter()
        train_time = time_end - time_start
        print('running time %ss' % train_time)
        X, test_time = self.evaluation(max_iter)
        if is_save is True:
            torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')
            sio.savemat(self.save_path + self.strX, {'X': X, 't1': train_time, 't2': test_time})
        pass


if __name__ == '__main__':
    ndata, nratio, nsnr = 0, 8, 2
    stage, edm_num = 3, 80  # pavia: 3, 80; ksc: 3, 80; dc: 3, 30; UH: 3, 30
    blind = Blind(ndata=ndata, nratio=nratio, nsnr=nsnr, blind=True)
    blind.train()
    blind.get_save_result()
    net = MIAE(ndata=ndata, nratio=nratio, nsnr=nsnr, psf=blind.psf, srf=blind.srf, stage=stage, edm_num=edm_num)
    net.train(verb=True, is_save=False)
    pass
