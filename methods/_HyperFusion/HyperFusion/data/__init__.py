#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xlrd
import numpy as np
import os
import torch
import importlib
import glob
import scipy.io as io
from .dataset import Dataset


sp_root_path = "../data/spectral_response/"
estimated_R_root_path = '../data/EstimatedR'


def get_spectral_response(data_name):
    xls_path = os.path.join(sp_root_path, data_name + '.xls')
    if not os.path.exists(xls_path):
        raise Exception("spectral response path does not exist")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]

    num_cols = table.ncols
    # import ipdb
    # ipdb.set_trace()
    cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(1,num_cols)]

    sp_data = np.concatenate(cols_list, axis=1)
    sp_data = sp_data / (sp_data.sum(axis=0))

    return sp_data

def get_spectral_R_estimated(data_name, sigma):
    path_list = glob.glob(os.path.join(estimated_R_root_path,'*.mat'))
    path_list_lower = [filename.lower() for filename in path_list]
    for index, path in enumerate(path_list_lower):
        if sigma > 1:
            sigma = int(sigma)
        if path.find(data_name) >=0 and path.find(str(sigma)) >=0:
            estimated_R = io.loadmat(path_list[index])['R'].transpose(1,0)
            break
    return estimated_R

def create_dataset(arg, sp_matrix, isTRain):
    dataset_instance = Dataset(arg, sp_matrix, isTRain)
    return dataset_instance

def get_sp_range(sp_matrix):
    HSI_bands, MSI_bands = sp_matrix.shape

    assert(HSI_bands>MSI_bands)
    sp_range = np.zeros([MSI_bands,2])
    for i in range(0,MSI_bands):
        index_dim_0, index_dim_1 = np.where(sp_matrix[:,i].reshape(-1,1)>0)
        sp_range[i,0] = index_dim_0[0]
        sp_range[i,1] = index_dim_0[-1]
    return sp_range
class DatasetDataLoader():
    def init(self, arg, isTrain=True):
        self.sp_matrix = get_spectral_response(arg.data_name)
        # self.estimated_R = get_spectral_R_estimated(arg.data_name, arg.sigma)
        self.sp_range = get_sp_range(self.sp_matrix)
        self.dataset = create_dataset(arg, self.sp_matrix, isTrain)
        self.hsi_channels = self.dataset.hsi_channels
        self.msi_channels = self.dataset.msi_channels
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=arg.batchsize if isTrain else 1,
                                                      shuffle=arg.isTrain if isTrain else False,
                                                      num_workers=arg.nThreads if arg.isTrain else 0)
    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def get_dataloader(arg, isTrain=True):
    instant_dataloader = DatasetDataLoader()
    instant_dataloader.init(arg, isTrain)
    return instant_dataloader
