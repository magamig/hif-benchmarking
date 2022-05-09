from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

import logging 
import numpy as np
import os


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_logger(file_dir): 
    """Print the results in the log file."""
    logger = logging.getLogger() 
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S") 
    fhandler.setFormatter(formatter) # set format
    logger.addHandler(fhandler) 
    logger.setLevel(logging.INFO) 
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    """Save the checkpoint."""
    state = {
            'epoch': epoch,
            'iter': iteration,
            'state_dict': model.state_dict(), 
            'optimizer' : optimizer.state_dict(), 
            }
    
    torch.save(state, os.path.join(model_path, 'ssfsr_9layers_epoch%d.pkl' %(epoch)))
    # torch.save(model,'xxx.pth')

def record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()    
    loss_csv.close

