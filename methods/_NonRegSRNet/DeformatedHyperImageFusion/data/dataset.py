#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: zhengke
@Date: 2020-07-19 21:16:56
@LastEditors: zhengke
@LastEditTime: 2020-07-20 08:31:58
@Description: dataset class
@FilePath: \DeformatedHyperImageFusion\data\dataset.py
"""


import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
import random
from scipy.interpolate import interp1d
import cv2
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    """
    @description: numpy data transform to tensor data
    @param {training options,
            data dict} 
    @return: tensor data
    """

    def __init__(self, args, data_dict):
        super(Dataset, self).__init__()

        self.args = args
        self.data_dict = data_dict
        self.hsi_channels, self.msi_channels = data_dict.SRF.shape

    def __getitem__(self, index):
        img_patch = self.data_dict.HrHSI
        img_lr = self.data_dict.LrHSI
        img_msi = self.data_dict.HrMSI

        img_name = self.data_dict.data_dir_name

        img_tensor_lr = torch.from_numpy(img_lr.transpose(2, 0, 1).copy()).float()
        img_tensor_hr = torch.from_numpy(img_patch.transpose(2, 0, 1).copy()).float()
        img_tensor_rgb = torch.from_numpy(img_msi.transpose(2, 0, 1).copy()).float()

        return {"lhsi": img_tensor_lr, "hmsi": img_tensor_rgb, "hhsi": img_tensor_hr, "name": img_name}

    def __len__(self):
        return 1

