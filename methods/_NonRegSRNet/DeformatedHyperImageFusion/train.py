#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: zhengke
@Date: 2020-07-19 21:16:56
@LastEditTime: 2020-07-20 05:50:04
@LastEditors: zhengke
@Description: main function for training
@FilePath: \DeformatedHyperImageFusion\train.py
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

    train_dataloader = get_dataloader(train_opt)
    dataset_size = len(train_dataloader)

    train_model = create_model(train_opt, train_dataloader.data_dict)
    train_model.setup(train_opt)

    visualizer = Visualizer(train_opt, train_dataloader.data_dict.wl)

    total_steps = 0

    for epoch in range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        train_psnr_list = []

        for i, data in enumerate(train_dataloader):

            iter_start_time = time.time()
            total_steps += train_opt.batchsize
            epoch_iter += train_opt.batchsize

            visualizer.reset()

            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)

            hues.info(
                "[{}/{} in {}/{}]".format(
                    i, dataset_size // train_opt.batchsize, epoch, train_opt.niter + train_opt.niter_decay,
                )
            )

            train_psnr = train_model.cal_psnr()
            train_psnr_list.append(train_psnr)

            if epoch % train_opt.print_freq == 0:
                losses = train_model.get_current_losses()
                t = (time.time() - iter_start_time) / train_opt.batchsize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t)
                if train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, train_opt, losses)
                    visualizer.display_current_results(
                        train_model.get_current_visuals(), train_model.get_image_name(), epoch, True, win_id=[1],
                    )

                    visualizer.plot_spectral_lines(
                        train_model.get_current_visuals(),
                        train_model.get_image_name(),
                        visual_corresponding_name=train_model.get_visual_corresponding_name(),
                        win_id=[2, 3],
                    )
                    visualizer.plot_psnr_sam(
                        train_model.get_current_visuals(),
                        train_model.get_image_name(),
                        epoch,
                        float(epoch_iter) / dataset_size,
                        train_model.get_visual_corresponding_name(),
                    )

                    visualizer.plot_lr(train_model.get_LR(), epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time,)
        )

        train_model.update_learning_rate()

    train_model.savePSFweight()
    # train_model.save_networks(train_opt.niter + train_opt.niter_decay)
    train_model.saveAbundance()
    train_model.saveDeformationField()
    train_model.saveReconstructionImage()
