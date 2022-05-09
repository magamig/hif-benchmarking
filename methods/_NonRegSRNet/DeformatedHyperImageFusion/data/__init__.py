#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: zhengke
@Date: 2020-07-19 21:16:56
@LastEditors: zhengke
@LastEditTime: 2020-07-20 06:16:37
@Description: data module inital function
@FilePath: \DeformatedHyperImageFusion\data\__init__.py
"""

import xlrd
import numpy as np
import os
import torch
import importlib
import glob
import scipy.io as io
from .dataset import Dataset
import ipdb


root_path = "../data/"


class DataDict:
    def __init__(self, args):
        """
        @description: read mat file
        @param {trainign options} 
        @return: data dict
        """
        self.args = args
        self.data_dir_name = self.args.data_dir
        self.data_path = os.path.join(root_path, self.args.data_dir, "data.mat")
        # fdx fdy HrHSI HrMSI LrHSI PSF_sigma Scale_factor SRF x y
        mat_dict = io.loadmat(self.data_path)
        # self.fdx = mat_dict["fdx"]
        # self.fdy = mat_dict["fdy"]
        self.HrHSI = mat_dict["HrHSI"]
        self.HrMSI = mat_dict["HrMSI"]
        self.LrHSI = mat_dict["LrHSI"]
        self.PSF_sigma = mat_dict["PSF_sigma"]
        self.Scale_factor = mat_dict["Scale_factor"]
        self.SRF = mat_dict["SRF"]
        self.srf_range = self.get_srf_range(self.SRF)
        # self.x = mat_dict["x"]
        # self.y = mat_dict["y"]
        self.wl = mat_dict["wl"]

        self.hsi_channels = self.HrHSI.shape[2]
        self.msi_channels = self.HrMSI.shape[2]

    def get_srf_range(self, sp_matrix):
        """
        @description: transform srf to srf_range(the began and end spectral bands of msi image 
                                                corresponding to hsi's index bands)
        @param {srf} 
        @return: srf_range
        """
        HSI_bands, MSI_bands = sp_matrix.shape
        assert HSI_bands > MSI_bands
        sp_range = np.zeros([MSI_bands, 2])
        # import ipdb
        # ipdb.set_trace()
        for i in range(0, MSI_bands):
            index_dim_0, index_dim_1 = np.where(sp_matrix[:, i].reshape(-1, 1) > 0)
            sp_range[i, 0] = index_dim_0[0]
            sp_range[i, 1] = index_dim_0[-1]
        return sp_range


class DatasetDataLoader:
    """
    @description: dataloader class
    @param {training options} 
    @return: tensor data
    """

    def __init__(self, arg):
        self.data_dict = DataDict(arg)
        self.dataset = Dataset(arg, self.data_dict)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=arg.batchsize, shuffle=False, num_workers=0
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def get_dataloader(arg):
    """
    @description: like function name
    @param {training options} 
    @return: dataloader instance
    """
    instant_dataloader = DatasetDataLoader(arg)
    return instant_dataloader
