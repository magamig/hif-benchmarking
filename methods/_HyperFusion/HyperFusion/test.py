#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The framework of traing process
~~~~~
Before python train.py, please ensure running the "python -m visdom.server -port=xxx" where xxx is the port assign in options
"""


import torch
import torch.nn as nn
import time
import numpy as np
import hues
import os
from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer


if __name__ == "__main__":

    train_opt = TrainOptions().parse()
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)

    train_model.setup(train_opt)

    train_model.load_networks(train_opt.which_epoch)

    for i, data in enumerate(train_dataloader):


        train_model.set_input(data, False)

        train_model.saveAbundance()
