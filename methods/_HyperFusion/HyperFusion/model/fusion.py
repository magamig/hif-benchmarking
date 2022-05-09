#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
import itertools
from . import network
from .base_model import BaseModel
import hues
import os
import numpy as np
import skimage.measure as ski_measure
import scipy.io as io



class Fusion(BaseModel):
    def name(self):
        return 'FusionGan'

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):

        parser.set_defaults(no_dropout=True)
        if isTrain:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for lr_lr')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for msi_msi    beta')
            parser.add_argument('--lambda_C', type=float, default=1.0, help='weight for msi_s_lr   alpha')
            parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for sum2one    mu')
            parser.add_argument('--lambda_E', type=float, default=1.0, help='weight for sparse     nu')
            parser.add_argument('--lambda_F', type=float, default=1.0, help='weight for lrmsi      gamma')
            parser.add_argument('--lambda_G', type=float, default=0.0, help='non')
            parser.add_argument('--lambda_H', type=float, default=0.0, help='non')
            parser.add_argument('--num_theta', type=int, default=128)
            parser.add_argument('--n_res', type=int, default=3)
            # parser.add_argument('--avg_crite', type=str, default='No')
            parser.add_argument('--avg_crite', action="store_true")
            # parser.add_argument('--ThetaModel', type=str, default='Normal', help='Normal Dirichlet')
            # parser.add_argument('--pixelwiseloss', type=str, default='l1', help='l2, l1')
            # parser.add_argument('--useGan', type=str, default='Yes', help='Yes No')
            parser.add_argument('--useGan', action="store_true")
            # parser.add_argument('--isRealFusion', type=str, default='No', help='Yes No')
            # parser.add_argument('--isCalSP', type=str, default='No')
            parser.add_argument('--isCalSP', action="store_true")
            # parser.add_argument('--all_band_calSP', type=str, default='No', help='is using all hr bands cal SP')
            # parser.add_argument("--useSoftmax", type=str, default='Yes')
            parser.add_argument("--useSoftmax", action='store_false')
        return parser


    def initialize(self, opt, hsi_channels, msi_channels, sp_matrix, sp_range):
        

        BaseModel.initialize(self, opt)

        self.opt = opt

        self.visual_names = ['real_lhsi', 'rec_lr_lr']

        num_s = self.opt.num_theta

        # net generate abundance (encoder for msi)
        self.net_MSI2S = network.define_msi2s(input_ch=msi_channels, output_ch=num_s, gpu_ids=self.gpu_ids, n_res=opt.n_res,
                                                useSoftmax=opt.useSoftmax)
        # shared endmember (also represents decoder)
        self.net_s2img = network.define_s2img(input_ch=num_s, output_ch=hsi_channels, gpu_ids=self.gpu_ids)
        # encoder for hsi
        self.net_LR2s = network.define_lr2s(input_ch=hsi_channels, output_ch=num_s, gpu_ids=self.gpu_ids, n_res=opt.n_res,
                                                useSoftmax=opt.useSoftmax)
        # define psf function
        self.net_PSF = network.define_psf(scale=opt.scale_factor,gpu_ids=self.gpu_ids)
        # reconstruct msi image from target HRHSI image
        self.net_G_HR2MSI = network.define_hr2msi(args=self.opt,
                                                    hsi_channels=hsi_channels,
                                                    msi_channels=msi_channels,
                                                    sp_matrix=sp_matrix,
                                                    sp_range=sp_range,
                                                    gpu_ids=self.gpu_ids)
        
        # LOSS
        if self.opt.avg_crite == False:
            self.criterionL1Loss = torch.nn.L1Loss(size_average=False).to(self.device)
        else:
            self.criterionL1Loss = torch.nn.L1Loss(size_average=True).to(self.device)
        self.criterionPixelwise = self.criterionL1Loss
        self.criterionSumToOne = network.SumToOneLoss().to(self.device)
        self.criterionSparse = network.SparseKLloss().to(self.device)

        self.model_names = ['MSI2S', 's2img', 'LR2s', 'PSF', 'G_HR2MSI']

        self.setup_optimizers()

        self.visual_corresponding_name = {}

    def setup_optimizers(self, lr=None):
        if lr == None:
            lr = self.opt.lr
        else:
            isinstance(lr, float)
            lr = lr
        self.optimizers = []
        # 0.5
        self.optimizer_G_MSI2S = torch.optim.Adam(itertools.chain(self.net_MSI2S.parameters()),
                                            lr=lr*0.5,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_MSI2S)
        self.optimizer_G_s2img = torch.optim.Adam(itertools.chain(self.net_s2img.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_s2img)
        self.optimizer_G_LR2s = torch.optim.Adam(itertools.chain(self.net_LR2s.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_LR2s)
        # 0.2
        self.optimizer_G_PSF = torch.optim.Adam(itertools.chain(self.net_PSF.parameters()),
                                            lr=lr*0.2,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_PSF)
        
        if self.opt.isCalSP == True:
            # 0.2
            self.optimizer_G_HR2MSI = torch.optim.Adam(itertools.chain(self.net_G_HR2MSI.parameters()),
                                                       lr=lr*0.2,betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G_HR2MSI)


    def set_input(self, input, isTrain=True):
        if isTrain:
            self.real_lhsi = Variable(input['lhsi'], requires_grad=True).to(self.device)
            self.real_hmsi = Variable(input['hmsi'], requires_grad=True).to(self.device)
            # if self.opt.isRealFusion == 'No':
            self.real_hhsi = Variable(input['hhsi'], requires_grad=True).to(self.device)

        else:
            with torch.no_grad():
                self.real_lhsi = Variable(input['lhsi'], requires_grad=False).to(self.device)
                self.real_hmsi = Variable(input['hmsi'], requires_grad=False).to(self.device)
                # if self.opt.isRealFusion == 'No':
                self.real_hhsi = Variable(input['hhsi'], requires_grad=False).to(self.device)

        self.image_name = input['name']

        self.real_input = input

    def forward(self):
        # first lr process
        self.rec_lr_s = self.net_LR2s(self.real_lhsi)
        self.rec_lr_lr = self.net_s2img(self.rec_lr_s)
        #second msi process
        self.rec_msi_s = self.net_MSI2S(self.real_hmsi)
        self.rec_msi_hr = self.net_s2img(self.rec_msi_s)
        self.rec_msi_msi = self.net_G_HR2MSI(self.rec_msi_hr)
        # third msi s lr
        self.rec_msi_lrs = self.net_PSF(self.rec_msi_s)
        self.rec_msi_lrs_lr = self.net_s2img(self.rec_msi_lrs)
        # four hr-msi-->psf-->lr-msi == lr-hsi-->sp-->lr-msi
        self.rec_lrhsi_lrmsi = self.net_G_HR2MSI(self.real_lhsi)
        self.rec_hrmsi_lrmsi = self.net_PSF(self.real_hmsi)

        self.visual_corresponding_name['real_lhsi'] = 'rec_lr_lr'
        self.visual_corresponding_name['real_hmsi'] = 'rec_msi_msi'
        # if self.opt.isRealFusion == 'No':
        self.visual_corresponding_name['real_hhsi'] = 'rec_msi_hr'


    def backward_joint(self, epoch):
        # lr
        self.loss_lr_pixelwise = self.criterionPixelwise(self.real_lhsi, self.rec_lr_lr) * self.opt.lambda_A
        self.loss_lr_s_sumtoone = self.criterionSumToOne(self.rec_lr_s) * self.opt.lambda_D
        self.loss_lr_sparse = self.criterionSparse(self.rec_lr_s) * self.opt.lambda_E
        self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_s_sumtoone + self.loss_lr_sparse
        # msi
        self.loss_msi_pixelwise = self.criterionPixelwise(self.real_hmsi, self.rec_msi_msi) * self.opt.lambda_B
        self.loss_msi_s_sumtoone = self.criterionSumToOne(self.rec_msi_s) * self.opt.lambda_D
        self.loss_msi_sparse = self.criterionSparse(self.rec_msi_s) * self.opt.lambda_E
        self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_s_sumtoone + self.loss_msi_sparse
        # PSF
        self.loss_msi_ss_lr =  self.criterionPixelwise(self.real_lhsi, self.rec_msi_lrs_lr) * self.opt.lambda_C
        # lrmsi
        self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_lrhsi_lrmsi, self.rec_hrmsi_lrmsi) * self.opt.lambda_F

        self.loss_joint = self.loss_lr  + self.loss_msi  + self.loss_msi_ss_lr + self.loss_lrmsi_pixelwise

        self.loss_joint.backward(retain_graph=True)

    def optimize_joint_parameters(self, epoch):
        self.loss_names = ["lr_pixelwise", 'lr_s_sumtoone', 'lr_sparse', 'lr',
                           'msi_pixelwise','msi_s_sumtoone','msi_sparse','msi',
                           'msi_ss_lr', 'lrmsi_pixelwise']
        

        self.visual_names = ['real_lhsi', 'rec_lr_lr', 'real_hmsi','rec_msi_msi','real_hhsi','rec_msi_hr']
        # if self.opt.isRealFusion == 'Yes':
        #     self.visual_names = ['real_lhsi', 'rec_lr_lr', 'real_hmsi','rec_msi_msi','rec_msi_hr']

        
        # self.set_requires_grad([self.net_G_s2img,self.net_G_LR2s,self.net_G_MSI2S,self.net_G_PSF, self.net_G_HR2MSI], True)
        self.forward()
        self.optimizer_G_LR2s.zero_grad()
        self.optimizer_G_s2img.zero_grad()
        self.optimizer_G_MSI2S.zero_grad()
        self.optimizer_G_PSF.zero_grad()
        if self.opt.isCalSP == 'Yes':
            self.optimizer_G_HR2MSI.zero_grad()
        self.backward_joint(epoch)
        self.optimizer_G_LR2s.step()
        self.optimizer_G_s2img.step()
        self.optimizer_G_MSI2S.step()
        self.optimizer_G_PSF.step()
        if self.opt.isCalSP == 'Yes':
            self.optimizer_G_HR2MSI.step()

        # clipper_nonzero = network.NonZeroClipper()
        # self.net_G_s2img.apply(clipper_nonzero)
        cliper_zeroone = network.ZeroOneClipper()
        self.net_PSF.apply(cliper_zeroone)
        self.net_s2img.apply(cliper_zeroone)
        if self.opt.isCalSP == 'Yes':
            cliper_sumtoone = network.SumToOneClipper()
            self.net_G_HR2MSI.apply(cliper_sumtoone)


        

    def savePSFweight(self):
        save_np = self.net_PSF.module.net.weight.data.cpu().numpy().reshape(self.opt.scale_factor,self.opt.scale_factor)
        # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'rec_psf_weight.npy')
        # np.save(save_path, save_np)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'rec_psf_weight.mat')
        io.savemat(save_path,{'psf_weight':save_np})

    def saveAbundance(self):
        self.forward()

        LHSI_A_a = self.rec_lr_s.data.cpu().numpy()
        # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'LHSI_A_a.npy')
        # np.save(save_path, LHSI_A_a)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_lhsi_a.mat')
        io.savemat(save_path,{'abundance_lhsi_a':LHSI_A_a})

        HMSI_A = self.rec_msi_s.data.cpu().numpy()
        # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'HMSI_A.npy')
        # np.save(save_path, HMSI_A)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_hmsi.mat')
        io.savemat(save_path,{'abundance_hmsi':HMSI_A})

        LHSI_A_b = self.rec_msi_lrs.data.cpu().numpy()
        # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'LHSI_A_b.npy')
        # np.save(save_path, LHSI_A_b)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_lhsi_b.mat')
        io.savemat(save_path,{'abundance_lhsi_b':LHSI_A_b})

    def get_visual_corresponding_name(self):
        return self.visual_corresponding_name

    def cal_psnr(self):
        real_hsi = self.real_hhsi.data.cpu().float().numpy()[0]
        rec_hsi = self.rec_msi_hr.data.cpu().float().numpy()[0]
        return self.compute_psnr(real_hsi, rec_hsi)

    def compute_psnr(self, img1, img2):
        assert img1.ndim == 3 and img2.ndim ==3

        img_c, img_w, img_h = img1.shape
        ref = img1.reshape(img_c, -1)
        tar = img2.reshape(img_c, -1)
        msr = np.mean((ref - tar)**2, 1)
        max2 = np.max(ref,1)**2
        psnrall = 10*np.log10(max2/msr)
        out_mean = np.mean(psnrall)
        return out_mean

    def get_sp_weight(self):
        if self.opt.isCalSP == 'Yes':
            parameter_list = [i.view(1,-1) for i in self.net_G_HR2MSI.parameters()]
            print(parameter_list[0])

    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]['lr'] * 2 * 1000
        return lr
