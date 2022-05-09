# coding: utf-8
# Script for training TSFN
#
# Reference: 
# Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation
# Xiuheng Wang, Jie Chen, Qi Wei, Cédric Richard
#
# 2019/05
# Implemented by
# Xiuheng Wang
# xiuheng.wang@mail.nwpu.edu.cn

from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import time
import argparse

from dataset import Dataset_cave_train, Dataset_cave_val
from model import Net
from utils import AverageMeter,initialize_logger,save_checkpoint,record_loss
from loss import MyLoss

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='CAVE database superresolution:')
# model configuration
parser.add_argument('--HSI_num_residuals', type=int, default=6, help="Set numbers of ResBlock of HSI branch")
parser.add_argument('--RGB_num_residuals', type=int, default=6, help="Set numbers of ResBlock of RGB branch")
# model storage path
parser.add_argument('--model_path', type=str, default='./models/', help='Set model storage path')

args = parser.parse_args()

def main():
    
    cudnn.benchmark = True 
    train_data = Dataset_cave_train('./data/train')
    print('number of train data: ', len(train_data))
    val_data = Dataset_cave_val('./data/test')
    print('number of validate data: ', len(val_data))

    # Model               
    model = Net(HSI_num_residuals=args.HSI_num_residuals, RGB_num_residuals=args.RGB_num_residuals)

    # multi-GPU setup
    device = torch.device("cuda:0, 1" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)

    model=model.to(device=device, dtype=torch.float) # float32
    model.apply(weights_init_kaiming)

    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 501
    init_lr = 0.0002
    iteration = 0

    criterion = MyLoss() 
    optimizer=torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, 
    #     threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    model_path = args.model_path 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss.csv'), 'w+') 
    
    log_dir = os.path.join(model_path,'train.log')
    logger = initialize_logger(log_dir)
    
    # Resume
    resume_file = ''   
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            # start_epoch = checkpoint['epoch']
            # iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch+1, end_epoch):

        train_data = Dataset_cave_train('./data/train')
        train_data_loader = DataLoader(dataset=train_data, 
                                   num_workers=8,  
                                   batch_size=16,
                                   shuffle=True,
                                   pin_memory=True,
                                   )
        
        val_data = Dataset_cave_val('./data/test')
        val_data_loader = DataLoader(dataset=val_data,
                            num_workers=8, 
                            batch_size=16,
                            shuffle=False,
                            pin_memory=True)
        
        start_time = time.time()
        train_loss, iteration = train(train_data_loader, model, criterion, optimizer, iteration, device)
        
        val_loss = validate(val_data_loader, model, criterion, device)

        # Save model
        if epoch % 100 ==0:
            save_checkpoint(model_path, epoch, iteration, model, optimizer)
        

        # # Update learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        # scheduler.step(val_loss)

        # print loss 
        end_time = time.time()
        epoch_time = end_time - start_time
        print ("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, val_loss))
        # save loss
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, val_loss)   # 调用record_方法：将epoch等6个指标写到csv文件中  
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, val_loss))
        

def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

# Training 
def train(train_data_loader, model, criterion, optimizer, iteration, device):
    losses = AverageMeter() 
    for i, (data_hsi, data_rgb, label) in enumerate(train_data_loader):

        data_hsi = data_hsi.to(device=device) 
        data_rgb = data_rgb.to(device=device)
        label = label.to(device=device)
        
        iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(data_hsi, data_rgb)
        
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        #  record loss
        losses.update(loss.item()) 
            
    return losses.avg, iteration 

# Validate
def validate(val_data_loader, model, criterion, device):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (data_hsi, data_rgb, label) in enumerate(val_data_loader):

        data_hsi = data_hsi.to(device=device)
        data_rgb = data_rgb.to(device=device)
        label = label.to(device=device)

        # compute output
        output = model(data_hsi, data_rgb)
        loss = criterion(output, label)

        #  record loss
        losses.update(loss.item())

    return losses.avg


if __name__ == '__main__':
    main()
