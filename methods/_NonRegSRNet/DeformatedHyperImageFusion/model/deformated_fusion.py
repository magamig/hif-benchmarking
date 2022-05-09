#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: zhengke
@Date: 2020-07-19 21:16:56
@LastEditors: zhengke
@LastEditTime: 2020-07-20 06:18:46
@Description: 
@FilePath: \DeformatedHyperImageFusion\model\fusion_gan.py
"""

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
import argparse
import ast


def str2bool(v):
    """
    @description: str to bool 
    @param {str} 
    @return: bool 
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DeformatedFusion(BaseModel):
    def name(self):
        return "DeformatedFusion"

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):
        """
        @description: parser operation about model, especial for hyper-parameters
        @param {training parser options} 
        @return: parser
        """
        parser.set_defaults(no_dropout=True)
        if isTrain:
            parser.add_argument("--num_rho", type=int, default=128, help="number of endmembers")
            parser.add_argument("--alpha", type=float, default=1.0, help="msi reconstruction loss")
            parser.add_argument("--beta", type=float, default=1.0, help="sum2one loss")
            parser.add_argument("--gamma", type=float, default=1.0, help="vox loss")
            parser.add_argument("--delta", type=float, default=1.0, help="recovered pixel loss")
            parser.add_argument("--theta", type=float, default=1.0, help="regularization for SRF")
            parser.add_argument("--mu", type=float, default=1.0, help="balance ncc and grad")
            parser.add_argument("--lambda_G", type=float, default=1.0, help="non")
            parser.add_argument("--lambda_H", type=float, default=0.0, help="non")
            # parser.add_argument("--n_res", type=int, default=3)

            parser.add_argument(
                "--avg_crite", type=ast.literal_eval, dest="avg_crite", default=False, help="l1 loss mode.",
            )

            parser.add_argument(
                "--isCalSP", type=ast.literal_eval, dest="isCalSP", default=True, help="srf mode.",
            )

            parser.add_argument(
                "--useClamp", type=ast.literal_eval, dest="useClamp", default=True, help="Activation mode.",
            )

        return parser

    def initialize(self, opt, data_dict):

        hsi_channels = data_dict.hsi_channels
        msi_channels = data_dict.msi_channels
        sp_matrix = data_dict.SRF
        sp_range = data_dict.srf_range
        lr_size = data_dict.LrHSI.shape[1:-1]

        self.scale_factor = data_dict.Scale_factor[0][0]

        BaseModel.initialize(self, opt)

        self.opt = opt

        self.visual_names = ["real_LrHsi", "rec_LrA_LrHsi"]

        num_p = self.opt.num_rho

        # define network modules
        self.net_MSI2A = network.define_msi2Abundance(
            input_ch=msi_channels, output_ch=num_p, gpu_ids=self.gpu_ids, useClamp=opt.useClamp,
        )

        self.net_A2Img = network.define_A2img(input_ch=num_p, output_ch=hsi_channels, gpu_ids=self.gpu_ids)

        self.net_HSI2A = network.define_HSI2A(
            input_ch=hsi_channels, output_ch=num_p, gpu_ids=self.gpu_ids, useClamp=opt.useClamp,
        )

        self.net_PSF = network.define_psf(scale=data_dict.Scale_factor[0][0], gpu_ids=self.gpu_ids)

        self.net_HSI2MSI = network.define_hsi2msi(
            args=self.opt,
            hsi_channels=hsi_channels,
            msi_channels=msi_channels,
            sp_matrix=sp_matrix,
            sp_range=sp_range,
            gpu_ids=self.gpu_ids,
        )

        # define f_theta() and STN 
        self.net_Displacefiled = network.define_displacementfiled(2 * num_p, 2, gpu_ids=self.gpu_ids)
        self.net_Spacetransform = network.define_spatial_transform(lr_size)      

        # LOSS
        if not self.opt.avg_crite:
            self.criterionL1Loss = torch.nn.L1Loss(size_average=False).to(self.device)
        else:
            self.criterionL1Loss = torch.nn.L1Loss(size_average=True).to(self.device)
        self.criterionPixelwise = self.criterionL1Loss
        self.criterionSumToOne = network.SumToOneLoss().to(self.device)
        # self.criterionVoxMorpLoss = network.VoxMorphLoss(n=9, lamda=0.1).to(self.device)
        self.criterionNCC = network.cross_correlation_loss
        self.criterionGrad = network.gradient_loss

        self.criterionCC = network.cross_correlation_loss
        self.criterionSM = network.smooothing_loss
        self.criterionVoxloss = network.vox_morph_loss

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
        self.optimizer_MSI2A = torch.optim.Adam(
            itertools.chain(self.net_MSI2A.parameters()), lr=lr, betas=(0.9, 0.999),
        )
        self.optimizers.append(self.optimizer_MSI2A)
        self.optimizer_A2Img = torch.optim.Adam(itertools.chain(self.net_A2Img.parameters()), lr=lr, betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_A2Img)
        self.optimizer_HSI2A = torch.optim.Adam(itertools.chain(self.net_HSI2A.parameters()), lr=lr, betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_HSI2A)
        # 0.2
        self.optimizer_PSF = torch.optim.Adam(itertools.chain(self.net_PSF.parameters()), lr=lr, betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_PSF)

        self.optimizer_Displacementfiled = torch.optim.Adam(
            itertools.chain(self.net_Displacefiled.parameters()), lr=lr*0.5, betas=(0.9, 0.999),
        )
        self.optimizers.append(self.optimizer_Displacementfiled)

        if self.opt.isCalSP:
            
            self.optimizer_HSI2MSI = torch.optim.Adam(list(self.net_HSI2MSI.parameters()), lr=lr, betas=(0.9, 0.999))
            # self.optimizer_HSI2MSI = torch.optim.Adam(self.net_HSI2MSI.parameters(), lr=lr*10, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_HSI2MSI)

    def set_input(self, input, isTrain=True):
        if isTrain:
            self.real_LrHsi = Variable(input["lhsi"], requires_grad=True).to(self.device)
            self.real_HrMsi = Variable(input["hmsi"], requires_grad=True).to(self.device)
            self.real_HrHsi = Variable(input["hhsi"], requires_grad=True).to(self.device)

        else:
            with torch.no_grad():
                self.real_LrHsi = Variable(input["lhsi"], requires_grad=False).to(self.device)
                self.real_HrMsi = Variable(input["hmsi"], requires_grad=False).to(self.device)
                self.real_HrHsi = Variable(input["hhsi"], requires_grad=False).to(self.device)

        self.image_name = input["name"]

        self.real_input = input

    def forward(self):
        # first lr process
        self.rec_LrHsi_LrA = self.net_HSI2A(self.real_LrHsi)
        self.rec_LrA_LrHsi = self.net_A2Img(self.rec_LrHsi_LrA)
        # second msi process
        self.rec_Msi_HrA = self.net_MSI2A(self.real_HrMsi)
        self.rec_HrA_HrHsi = self.net_A2Img(self.rec_Msi_HrA)
        self.rec_HrHsi_Msi = self.net_HSI2MSI(self.rec_HrA_HrHsi)
        # third msi hra psf lra
        self.rec_HrA_LrA = self.net_PSF(self.rec_Msi_HrA)
        # registration
        #! the shape of displacementfiled is [batchsize, y, x, 2]
        self.rec_DisplacementField = self.net_Displacefiled(
            torch.cat((self.rec_LrHsi_LrA, self.rec_HrA_LrA), dim=1)
        ).permute(0, 2, 3, 1)
        #! the shape of recovered abundance is [batchsize, y, x, num_p]
        self.rec_RecoveredAbundance = self.net_Spacetransform(
            self.rec_LrHsi_LrA.permute(0, 2, 3, 1), self.rec_DisplacementField
        )
        #! the shape of recoveredLrHSI is [batchsize, channels, y, x]
        self.rec_RecoveredLrHSI = self.net_A2Img(self.rec_RecoveredAbundance.permute(0, 3, 1, 2))
        # four hr-msi-->psf-->lr-msi == lr-hsi-->sp-->lr-msi
        
        #! NON MEANING, DELETED
        # self.rec_LrHsi_LrMsi = self.net_HSI2MSI(self.real_LrHsi)
        # self.rec_Hrmsi_Lrmsi = self.net_PSF(self.real_HrMsi)

        self.visual_corresponding_name["real_LrHsi"] = "rec_LrA_LrHsi"
        self.visual_corresponding_name["real_HrMsi"] = "rec_HrHsi_Msi"
        self.visual_corresponding_name["real_HrHsi"] = "rec_HrA_HrHsi"

    def backward_joint(self, epoch):
        # lr
        self.loss_lr_pixelwise = self.criterionPixelwise(self.real_LrHsi, self.rec_LrA_LrHsi)
        self.loss_lr_A_sumtoone = self.criterionSumToOne(self.rec_LrHsi_LrA) * self.opt.beta

        self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_A_sumtoone
        # msi
        self.loss_msi_pixelwise = self.criterionPixelwise(self.real_HrMsi, self.rec_HrHsi_Msi) * self.opt.alpha
        self.loss_msi_A_sumtoone = self.criterionSumToOne(self.rec_Msi_HrA) * self.opt.beta

        self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_A_sumtoone
        # PSF
        self.loss_msi_A_lr = (
            self.criterionPixelwise(self.rec_RecoveredLrHSI, self.net_PSF(self.rec_HrA_HrHsi)) * self.opt.delta
        )

        # lrmsi
        # self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_LrHsi_LrMsi, self.rec_Hrmsi_Lrmsi) * self.opt.lambda_F

        # deformationmation loss
        self.loss_ncc = self.criterionCC(self.rec_RecoveredAbundance.permute(0, 3, 1, 2), self.rec_HrA_LrA, int(self.opt.mu))
        self.loss_grad = self.criterionSM(self.rec_DisplacementField.permute(0, 3, 1, 2))
        self.loss_voxmorphloss = (1 - self.loss_ncc) * self.opt.gamma + self.loss_grad * self.opt.theta 
        
        # ! new added smooth regularization for the weights of net_HSI2MSI
        # self.loss_reg_hsi2msi = 0
        # for param in self.net_HSI2MSI.parameters():
        #     #! L2 normalization
        #     dz = torch.abs(param[:, 1:, :, :] - param[:, :-1, :, :])
        #     self.loss_reg_hsi2msi += torch.sum(dz * dz) * self.opt.theta

        # self.loss_joint = (
        #     self.loss_lr + self.loss_msi + self.loss_msi_A_lr + self.loss_voxmorphloss  + self.loss_reg_hsi2msi
        # )
        self.loss_joint = (
            self.loss_lr + self.loss_msi + self.loss_msi_A_lr + self.loss_voxmorphloss 
        )

        self.loss_joint.backward(retain_graph=True)

    def optimize_joint_parameters(self, epoch):
        self.loss_names = [
            "lr_pixelwise",
            "lr_A_sumtoone",
            "lr",
            "msi_pixelwise",
            "msi_A_sumtoone",
            "msi",
            "msi_A_lr",
            "ncc",
            "grad",
            "voxmorphloss",
            # "reg_hsi2msi",
        ]

        self.visual_names = [
            "real_LrHsi",
            "rec_LrA_LrHsi",
            "real_LrHsi",
            "rec_RecoveredLrHSI",
            "real_HrMsi",
            "rec_HrHsi_Msi",
            "real_HrHsi",
            "rec_HrA_HrHsi",
        ]

        self.set_requires_grad(
            [self.net_A2Img, self.net_HSI2A, self.net_MSI2A, self.net_PSF, self.net_HSI2MSI, self.net_Displacefiled,],
            True,
        )
        self.forward()
        self.optimizer_HSI2A.zero_grad()
        self.optimizer_A2Img.zero_grad()
        self.optimizer_MSI2A.zero_grad()
        self.optimizer_PSF.zero_grad()
        self.optimizer_Displacementfiled.zero_grad()
        if self.opt.isCalSP:
            self.optimizer_HSI2MSI.zero_grad()
        self.backward_joint(epoch)
        self.optimizer_HSI2A.step()
        self.optimizer_A2Img.step()
        self.optimizer_MSI2A.step()
        self.optimizer_PSF.step()
        self.optimizer_Displacementfiled.step()
        if self.opt.isCalSP:
            self.optimizer_HSI2MSI.step()

        # clipper_nonzero = network.NonZeroClipper()
        # self.net_A2Img.apply(clipper_nonzero)
        cliper_zeroone = network.ZeroOneClipper()
        self.net_PSF.apply(cliper_zeroone)
        self.net_A2Img.apply(cliper_zeroone)
        if self.opt.isCalSP:
            cliper_sumtoone = network.SumToOneClipper()
            self.net_HSI2MSI.apply(cliper_sumtoone)

    def savePSFweight(self):
        save_np = self.net_PSF.module.net.weight.data.cpu().numpy().reshape(self.scale_factor, self.scale_factor)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "rec_psf_weight.mat")
        # np.save(save_path, save_np)
        io.savemat(save_path, {"psf_weight": save_np})

    def saveAbundance(self):
        self.forward()

        LHSI_A_a = self.rec_LrHsi_LrA.data.cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "Abundance_lhsi_a.mat")
        # np.save(save_path, LHSI_A_a)
        io.savemat(save_path, {"abundance_lhsi_a": LHSI_A_a})

        HMSI_A = self.rec_Msi_HrA.data.cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "Abundance_hmsi.mat")
        # np.save(save_path, HMSI_A)
        io.savemat(save_path, {"abundance_hmsi": HMSI_A})

        LHSI_A_b = self.rec_HrA_LrA.data.cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "Abundance_lhsi_b.mat")
        # np.save(save_path, LHSI_A_b)
        io.savemat(save_path, {"abundance_lhsi_b": LHSI_A_b})

    def saveDeformationField(self):
        self.forward()

        displacementfield = self.rec_DisplacementField.data.cpu().numpy()[0]
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "DisplacementField.mat")
        # np.save(save_path, displacementfield)
        io.savemat(save_path, {"displacement_field": displacementfield})

    def saveReconstructionImage(self):
        self.forward()

        rec_lr = self.rec_LrA_LrHsi.data.cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "ReconstructionLHSI.mat")
        io.savemat(save_path, {"reconstructionLHSI": rec_lr})

        rec_restored_lr = self.rec_RecoveredLrHSI.data.cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "RestoredLHSI.mat")
        io.savemat(save_path, {"restoredLHSI": rec_restored_lr})

        rec_msi = self.rec_HrHsi_Msi.data.cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "ReconstructionHMSI.mat")
        io.savemat(save_path, {"reconstructionHMSI": rec_msi})

    def get_visual_corresponding_name(self):
        return self.visual_corresponding_name

    def cal_psnr(self):
        real_hsi = self.real_HrHsi.data.cpu().float().numpy()[0]
        rec_hsi = self.rec_HrA_HrHsi.data.cpu().float().numpy()[0]
        return self.compute_psnr(real_hsi, rec_hsi)

    def compute_psnr(self, img1, img2):
        assert img1.ndim == 3 and img2.ndim == 3

        # n_bands = img1.shape[0]
        # psnr_list = [ski_measure.compare_psnr(img1[i,:,:], img2[i,:,:]) for i in range(n_bands)]
        # var_psnr = np.var(psnr_list)
        # mpsnr = np.mean(np.array(psnr_list))
        # return mpsnr, var_psnr
        img_c, img_w, img_h = img1.shape
        ref = img1.reshape(img_c, -1)
        tar = img2.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max2 = np.max(ref, 1) ** 2
        # import ipdb
        # ipdb.set_trace()
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean

    def get_sp_weight(self):
        if self.opt.isCalSP:
            parameter_list = [i.view(1, -1) for i in self.net_HSI2MSI.parameters()]
            print(parameter_list[0])

    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]["lr"] * 2 * 1000
        # import ipdb
        # ipdb.set_trace()
        return lr
