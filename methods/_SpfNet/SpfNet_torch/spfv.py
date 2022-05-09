import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import glob
import os
import math
import time
import scipy
import scipy.io as sio
import tensorly as tl
from SpfNet_torch.fusion import FusionNet, check_dir
from SpfNet_torch.utils import AverageMeter, toolkits, torchkits, to_pair

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pad_convt = lambda ker_size, stride: int(math.ceil((ker_size - stride) / 2))
out_pad_convt = lambda ker_size, stride: 2 * pad_convt(ker_size, stride) - (ker_size - stride)


class PieceDataset(data.Dataset):
    def __init__(self, data_path, num_piece=10000):
        super().__init__()
        self.data_path = data_path
        self.num_piece = num_piece
        self.num = 0
        for dir_name in os.listdir(self.data_path):
            if os.path.isdir(self.data_path + dir_name):
                self.num += len(glob.glob(pathname=self.data_path + dir_name + '/*.mat'))
        pass

    def __getitem__(self, item):
        it = item // self.num_piece
        file_path = self.data_path + str(it) + '/' + str(item - it * self.num_piece) + '.mat'
        mat = sio.loadmat(file_path)
        Y, Z, X, A = mat['LRHS'], mat['HRMS'], mat['HS'], mat['A']
        Y, Z, X, A = torch.tensor(Y), torch.tensor(Z), torch.tensor(X), torch.tensor(A)
        return Y, Z, X, A, item

    def __len__(self):
        return self.num


class DownResCNN(nn.Module):
    def __init__(self, feat_num, filter_size=3):
        super().__init__()
        self.feat_num = feat_num
        self.X1_net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(filter_size),
                      stride=to_pair(2),
                      padding=to_pair(pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )
        self.X2_net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(filter_size),
                      stride=to_pair(2),
                      padding=to_pair(pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )
        self.X3_net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(filter_size),
                      stride=to_pair(2),
                      padding=to_pair(pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )
        self.Y3_net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(filter_size),
                      stride=to_pair(1),
                      padding=to_pair(pad_convt(filter_size, 1))),
            nn.LeakyReLU(0.2)
        )
        self.Y2_net = nn.Sequential(
            nn.ConvTranspose2d(self.feat_num, self.feat_num,
                               kernel_size=to_pair(filter_size),
                               stride=to_pair(2),
                               padding=to_pair(pad_convt(filter_size, 2)),
                               output_padding=to_pair(out_pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )
        self.Y1_net = nn.Sequential(
            nn.ConvTranspose2d(self.feat_num, self.feat_num,
                               kernel_size=to_pair(filter_size),
                               stride=to_pair(2),
                               padding=to_pair(pad_convt(filter_size, 2)),
                               output_padding=to_pair(out_pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.feat_num, self.feat_num,
                               kernel_size=to_pair(filter_size),
                               stride=to_pair(2),
                               padding=to_pair(pad_convt(filter_size, 2)),
                               output_padding=to_pair(out_pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X):
        X1 = self.X1_net(X)
        X2 = self.X2_net(X1)
        X3 = self.X3_net(X2)
        Y3 = self.Y3_net(X3)
        Y3 = Y3 + X3
        Y2 = self.Y2_net(Y3)
        Y2 = Y2 + X2
        Y1 = self.Y1_net(Y2)
        Y1 = Y1 + X1
        output = self.net(Y1)
        return output + X


class UpResCNN(nn.Module):
    def __init__(self, feat_num, filter_size=3):
        super().__init__()
        self.feat_num = feat_num
        self.X1_net = nn.Sequential(
            nn.ConvTranspose2d(self.feat_num, self.feat_num,
                               kernel_size=to_pair(filter_size),
                               stride=to_pair(2),
                               padding=to_pair(pad_convt(filter_size, 2)),
                               output_padding=to_pair(out_pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )
        self.Y1_net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(filter_size),
                      stride=to_pair(1),
                      padding=to_pair(pad_convt(filter_size, 1))),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(filter_size),
                      stride=to_pair(2),
                      padding=to_pair(pad_convt(filter_size, 2))),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X):
        X1 = self.X1_net(X)
        Y1 = self.Y1_net(X1)
        Y1 = Y1 + X1
        output = self.net(Y1)
        return output + X


class SpatFusionNet(nn.Module):
    def __init__(self, feat_num, ratio, out_stages):
        super().__init__()
        self.feat_num = feat_num
        self.ratio = ratio
        self.out_stages = out_stages
        self.S1_net = nn.Sequential(
            nn.Conv2d(self.feat_num, self.feat_num,
                      kernel_size=to_pair(self.ratio + self.ratio // 2),
                      stride=to_pair(self.ratio),
                      padding=to_pair(pad_convt(self.ratio + self.ratio // 2, self.ratio))),
            nn.LeakyReLU(0.2)
        )
        self.S1_t_net = nn.Sequential(
            nn.ConvTranspose2d(self.feat_num, self.feat_num,
                               kernel_size=to_pair(self.ratio + self.ratio // 2),
                               stride=to_pair(self.ratio),
                               padding=to_pair(pad_convt(self.ratio + self.ratio // 2, self.ratio)),
                               output_padding=to_pair(out_pad_convt(self.ratio + self.ratio // 2, self.ratio)))
        )
        self.S3_net = nn.ModuleList([])
        for out_iter in range(self.out_stages):
            self.S3_net.append(
                nn.Sequential(
                    nn.Conv2d(self.feat_num * 4, self.feat_num,
                              kernel_size=to_pair(3),
                              stride=to_pair(1),
                              padding=to_pair(1)),
                    nn.LeakyReLU(0.2)
                )
            )

    def forward(self, S, AtA, AtY, RAtRA, RAtZ, V, D, V1, D1, out_iter):
        S1 = torch.einsum('nkl,nkhw->nlhw', AtA, S)
        S1 = self.S1_net(S1)
        S1 = S1 - AtY
        S1 = self.S1_t_net(S1)
        S2 = torch.einsum('nkl,nkhw->nlhw', RAtRA, S)
        S2 = S2 - RAtZ
        S3 = torch.cat([S1, S2, S - V - D, S - V1 - D1], dim=1)
        S3 = self.S3_net[out_iter](S3)
        S = S - S3
        return S


class SpfNet(nn.Module):
    def __init__(self, edm_num, hs_bands, ms_bands, ratio):
        super().__init__()
        self.out_stages = 5
        self.in_steps = 3
        self.edm_num = edm_num
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ratio = ratio
        R = torch.ones([self.ms_bands, self.hs_bands]) * (1.0 / self.hs_bands)
        self.R = nn.ParameterList(
            [nn.Parameter(R) for out_iter in range(self.out_stages)]
        )

        # self.S1_net = nn.ModuleList([])
        # self.S1_t_net = nn.ModuleList([])
        # self.S3_net = nn.ModuleList([])
        # for in_iter in range(self.in_steps):
        #     self.S1_net.append(
        #         nn.Sequential(
        #             nn.Conv2d(self.edm_num, self.edm_num,
        #                       kernel_size=to_pair(self.ratio + self.ratio // 2),
        #                       stride=to_pair(self.ratio),
        #                       padding=to_pair(pad_convt(self.ratio + self.ratio // 2, self.ratio))),
        #             nn.LeakyReLU(0.2)
        #         )
        #     )
        #     self.S1_t_net.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(self.edm_num, self.edm_num,
        #                                kernel_size=to_pair(self.ratio + self.ratio // 2),
        #                                stride=to_pair(self.ratio),
        #                                padding=to_pair(pad_convt(self.ratio + self.ratio // 2, self.ratio)),
        #                                output_padding=to_pair(out_pad_convt(self.ratio + self.ratio // 2, self.ratio)))
        #         )
        #     )
        #     for out_iter in range(self.out_stages):
        #         self.S3_net.append(
        #             nn.Sequential(
        #                 nn.Conv2d(self.edm_num * 4, self.edm_num,
        #                           kernel_size=to_pair(3),
        #                           stride=to_pair(1),
        #                           padding=to_pair(1)),
        #                 nn.LeakyReLU(0.2)
        #             )
        #         )
        #     pass

        self.spat_fusion_subnet = nn.ModuleList([])
        for in_iter in range(self.in_steps):
            self.spat_fusion_subnet.append(SpatFusionNet(self.edm_num, self.ratio, self.out_stages))

        self.down_res_cnn_net = nn.ModuleList([])
        self.up_res_cnn_net = nn.ModuleList([])
        self.X_net = nn.ModuleList([])
        for out_iter in range(self.out_stages):
            self.down_res_cnn_net.append(DownResCNN(self.hs_bands))
            self.up_res_cnn_net.append(UpResCNN(self.hs_bands))
            self.X_net.append(nn.Sequential(
                nn.Conv2d(self.hs_bands, self.hs_bands,
                          kernel_size=to_pair(3),
                          stride=to_pair(1),
                          padding=to_pair(1))
            ))
        self.init_weights(self)
        pass

    def forward(self, Y, Z, A):
        X = self.spat_fusion_net(Y, Z, A, out_stages=self.out_stages, in_steps=self.in_steps)
        return X

    def spat_fusion_net(self, Y, Z, A, out_stages=5, in_steps=3):
        AtA = torch.einsum('nbk,nbl->nkl', A, A)
        AtY = torch.einsum('nbk,nbhw->nkhw', A, Y)
        S = torch.zeros(size=(Z.size(0), self.edm_num, Z.size(2), Z.size(3)), dtype=torch.float).cuda()
        V, D = S, S
        V1, D1 = S, S
        Xout = torch.zeros(size=(Z.size(0), self.hs_bands, Z.size(2), Z.size(3)), dtype=torch.float).cuda()
        for out_iter in range(out_stages):
            RA = torch.matmul(self.R[out_iter], A)
            RAtRA = torch.einsum('nbk,nbl->nkl', RA, RA)
            RAtZ = torch.einsum('nbk,nbhw->nkhw', RA, Z)
            for in_iter in range(in_steps):
                S = self.spat_fusion_subnet[in_iter](S, AtA, AtY, RAtRA, RAtZ, V, D, V1, D1, out_iter)
                # S = self.spat_fusion_subnet(S, AtA, AtY, RAtRA, RAtZ, V, D, V1, D1, out_iter, in_iter)
            V = self.down_res_cnn_net[out_iter](S - D)
            D = D - (S - V)
            V1 = self.up_res_cnn_net[out_iter](S - D1)
            D1 = D1 - (S - V1)
            X = torch.einsum('nbk,nkhw->nbhw', A, S)
            X = self.X_net[out_iter](X)
            Xout = Xout + X
        return Xout / out_stages

    @staticmethod
    def init_weights(model, init_type='normal'):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # print(name)
                # for name, parameters in m.named_parameters():
                #     print(name, ':', parameters.size())
                if init_type == 'normal':
                    nn.init.kaiming_normal_(m.weight.data)
                elif init_type == 'uniform':
                    nn.init.kaiming_uniform_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        pass

    # def spat_fusion_subnet(self, S, AtA, AtY, RAtRA, RAtZ, V, D, V1, D1, out_iter, in_iter):
    #     S1 = torch.einsum('nkl,nkhw->nlhw', AtA, S)
    #     S1 = self.S1_net[in_iter](S1)
    #     S1 = S1 - AtY
    #     S1 = self.S1_t_net[in_iter](S1)
    #     S2 = torch.einsum('nkl,nkhw->nlhw', RAtRA, S)
    #     S2 = S2 - RAtZ
    #     S3 = torch.cat([S1, S2, S - V - D, S - V1 - D1], dim=1)
    #     S3 = self.S3_net[in_iter * self.out_stages + out_iter](S3)
    #     S = S - S3
    #     return S


class Spfv(FusionNet):
    def __init__(self, data_num, sim=True):
        super().__init__(data_num, sim)
        self.k = min(self.hs_bands, 31)
        self.prepare_dataset()
        self.lr = 1e-3
        self.model = SpfNet(self.k, self.hs_bands, self.ms_bands, self.ratio).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fun)
        self.train_dataset = PieceDataset(self.train_data_path)
        self.valid_dataset = PieceDataset(self.valid_data_path)
        # for name, parameters in self.model.named_parameters():
        #     print(name, ':', parameters.size())
        pass

    @staticmethod
    def lr_fun(epoch):
        if 0 <= epoch < 20:
            return 1
        if 20 <= epoch < 50:
            return 0.5
        if 50 <= epoch < 100:
            return 0.1
        if 100 <= epoch < 150:
            return 0.05
        return 0.01

    def prepare_dataset(self, num_piece=10000, valid_ratio=0.2, channel_first=True):
        if os.path.exists(self.valid_data_path + 'info.mat'):
            print('Train and Validation: xx piece exists!')
            num = sio.loadmat(self.train_data_path + 'info.mat')
            train_num = num['num']
            num = sio.loadmat(self.valid_data_path + 'info.mat')
            valid_num = num['num']
            return train_num, valid_num
        check_dir(self.train_data_path)
        check_dir(self.valid_data_path)
        train_num, valid_num = 0, 0
        for i in range(self.train_start, self.train_end + 1):
            mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
            X = mat['HS']
            Y = mat['LRHS']
            Z = mat['HRMS']
            rows, cols, _ = X.shape
            for x in range(0, rows - self.piece_size + 1, self.stride):
                for y in range(0, cols - self.piece_size + 1, self.stride):
                    label = X[x:x + self.piece_size, y:y + self.piece_size, :]
                    z_data = Z[x:x + self.piece_size, y:y + self.piece_size, :]
                    y_data = Y[x // self.ratio:(x + self.piece_size) // self.ratio,
                             y // self.ratio:(y + self.piece_size) // self.ratio, :]
                    A, _, _ = scipy.linalg.svd(tl.unfold(y_data, mode=2), full_matrices=False)
                    if channel_first is True:
                        label = np.transpose(label, axes=(2, 0, 1))
                        z_data = np.transpose(z_data, axes=(2, 0, 1))
                        y_data = np.transpose(y_data, axes=(2, 0, 1))
                    mat_dict = {'HS': label, 'LRHS': y_data, 'HRMS': z_data, 'A': A}
                    train_num, valid_num = self.train_valid_piece_save(train_num, valid_num, mat_dict,
                                                                       self.train_data_path, self.valid_data_path,
                                                                       num_piece, valid_ratio)
            print('Piece: %d has finished' % i)
        print('Piece done')
        sio.savemat(self.train_data_path + 'info.mat', {'num': train_num})
        sio.savemat(self.valid_data_path + 'info.mat', {'num': valid_num})
        return train_num, valid_num

    def train(self):
        train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=self.train_batch_size,
                                       shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = data.DataLoader(self.valid_dataset,
                                       batch_size=self.valid_batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        for epoch in range(0, self.epochs):
            t0 = time.perf_counter()
            self.train_epoch(train_loader)
            t1 = time.perf_counter()
            print('training', t1 - t0)
            self.validate(valid_loader, epoch)
            t2 = time.perf_counter()
            print('validation', t2 - t1)
        pass

    def train_epoch(self, loader):
        self.model.train()
        for i, (hsi, msi, label, A, item) in enumerate(loader):
            hsi, msi, label, A = hsi.cuda(), msi.cuda(), label.cuda(), A.cuda()
            target = self.model(hsi, msi, A)
            loss = torch.mean(torch.abs(target - label))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pass
        self.scheduler.step()

    def validate(self, loader, epoch):
        self.model.eval()
        psnr_val = AverageMeter()
        loss_val = AverageMeter()
        for i, (hsi, msi, label, A, item) in enumerate(loader):
            batch_size = item.size(0)
            hsi, msi, label, A = hsi.cuda(), msi.cuda(), label.cuda(), A.cuda()
            target = self.model(hsi, msi, A)
            loss = torch.mean(torch.abs(target - label))
            loss_val.update(torchkits.to_numpy(loss), batch_size)
            psnr = toolkits.psnr_fun(torchkits.to_numpy(target), torchkits.to_numpy(label), max_value=None)
            psnr_val.update(psnr, batch_size)
        lr = self.optimizer.param_groups[0]['lr']
        print('epoch: %s, lr: %s, psnr: %s, loss: %s' % (epoch, lr, psnr_val.avg, loss_val.avg))
        if psnr_val.avg > self.max_power:
            print('get a satisfying model')
            self.max_power = psnr_val.avg
            check_dir(self.model_save_path)
            torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')

    def test_piece(self, stride=None):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        if stride is None:
            stride = self.test_stride
        run_time = 0
        for i in range(self.test_start, self.test_end + 1):
            mat = sio.loadmat(self.sim_save_path + '%d.mat' % i)
            tY = mat['LRHS']
            tZ = mat['HRMS']
            output = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
            num_sum = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
            start = time.perf_counter()
            for x in range(0, tZ.shape[0] - self.piece_size + 1, stride):
                for y in range(0, tZ.shape[1] - self.piece_size + 1, stride):
                    end_x = x + self.piece_size
                    if end_x + stride > tZ.shape[0]:
                        end_x = tZ.shape[0]
                    end_y = y + self.piece_size
                    if end_y + stride > tZ.shape[1]:
                        end_y = tZ.shape[1]
                    itY = tY[x // self.ratio:end_x // self.ratio, y // self.ratio:end_y // self.ratio, :]
                    itZ = tZ[x:end_x, y:end_y, :]
                    itA, _, _ = scipy.linalg.svd(tl.unfold(itY, mode=2), full_matrices=False)
                    itA = itA[:, 0:self.k]
                    itY = toolkits.channel_first(itY)
                    itZ = toolkits.channel_first(itZ)
                    itA = np.expand_dims(itA, 0)
                    itY = torch.tensor(itY).cuda()
                    itZ = torch.tensor(itZ).cuda()
                    itA = torch.tensor(itA).cuda()
                    tmp = self.model(itY, itZ, itA)
                    tmp = torchkits.to_numpy(tmp)
                    tmp = toolkits.channel_last(tmp)
                    output[x:end_x, y:end_y, :] += tmp
                    num_sum[x:end_x, y:end_y, :] += 1
            output = output / num_sum
            end = time.perf_counter()
            run_time += end - start
            check_dir(self.output_save_path)
            sio.savemat(self.output_save_path + '%d.mat' % i, {'F': output, 'TM': end - start})
            print('test: %d has finished' % i)
        print('Time: %ss' % (run_time / (self.test_end - self.test_start + 1)))


if __name__ == '__main__':
    data_num = 0
    net = Spfv(data_num, sim=True)
    net.train()
    net.test_piece()
    net.show_final_result()
    pass
