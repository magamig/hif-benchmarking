# Code for "Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution"
# Author: Jin-Fan Hu

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DatasetFromHdf5
# from model import PanNet,summaries
from model import *
import numpy as np
import scipy.io as sio
import shutil
from torch.utils.tensorboard import SummaryWriter
import time

# ================== Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True  ###????????
cudnn.deterministic = True
# cudnn.benchmark = False
# ============= HYPER PARAMS(Pre-Defined) ==========#
# lr = 0.001
# epochs = 1010
# ckpt = 50
# batch_size = 32
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)   # optimizer 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

lr = 1e-3
epochs = 1000
ckpt_step = 50
batch_size = 3

model = MainNet().cuda()
# model = nn.DataParallel(model)
# model_path = "Weights/.pth"
# if os.path.isfile(model_path):
#     # Load the pretrained Encoder
#     model.load_state_dict(torch.load(model_path))
#     print('Network is Successfully Loaded from %s' % (model_path))
# from torchstat import stat
# stat(model, input_size=[(31, 16, 16), (3, 64, 64)])
# summaries(model, grad=True)
PLoss = nn.L1Loss(size_average=True).cuda()
# Sparse_loss = SparseKLloss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   # optimizer 1
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  # optimizer 2
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                              gamma=0.1)  # lr = lr* 1/gamma for each step_size = 180

# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
#     shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs  --host=127.0.0.1
# writer = SummaryWriter('./train_logs(model-Trans)/')

model_folder = "Trained_model/"
writer = SummaryWriter("train_logs/ "+model_folder)
def save_checkpoint(model, epoch):  # save model function

    model_out_path = model_folder + "{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr":lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader,start_epoch=0,RESUME=False):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=2)
    print('Start training...')

    if RESUME:
        path_checkpoint = model_folder+"{}.pth".format(500)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    time_s = time.time()
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            GT, LRHSI, HRMSI  = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

            optimizer.zero_grad()  # fixed

            output_HRHSI,UP_LRHSI,Highpass = model(LRHSI,HRMSI)
            time_e = time.time()
            Pixelwise_Loss =PLoss(output_HRHSI, GT)
            # Pixelwise_Loss  = PLoss(Rec_LRHSI, LRHSI) + PLoss(Rec_HRMSI,HRMSI)
            # Sparse = Sparse_loss(A) + Sparse_loss(A_LR)
            # ASC_loss = Sum2OneLoss(A) + Sum2OneLoss(A_LR)

            Myloss = Pixelwise_Loss
            epoch_train_loss.append(Myloss.item())  # save all losses into a vector for one epoch

            Myloss.backward()  # fixed
            optimizer.step()  # fixed

            if iteration % 10 == 0:
                # log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Myloss.item()))
                # for name, parameters in model.named_parameters():
                #     print(name, ':', parameters.size(), parameters)
            # for name, layer in model.named_parameters():
            #     # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
            #     writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)
        print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        print(time_e - time_s)
        if epoch % ckpt_step == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
        # ============Epoch Validate=============== #
        if epoch % 50== 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT,  LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                    output_HRHSI,UP_LRHSI,Highpass = model(
                        LRHSI, HRMSI)
                    time_e = time.time()
                    Pixelwise_Loss = PLoss(output_HRHSI, GT)
                    MyVloss = Pixelwise_Loss
                    epoch_val_loss.append(MyVloss.item())
            LRHSI1 = LRHSI[0, [10, 20, 30], ...].float().permute(1, 2, 0).cpu().numpy()
            axes[0, 0].imshow(LRHSI1)
            axes[0, 1].imshow(HRMSI[0, ...].permute(1, 2, 0).cpu().numpy())
            axes[1, 0].imshow(output_HRHSI[0, [10, 20, 30], ...].permute(1, 2, 0).cpu().detach().numpy())
            axes[1, 1].imshow(GT[0, [10, 20, 30], ...].permute(1, 2, 0).cpu().numpy())
            plt.pause(0.1)
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/loss', v_loss, epoch)
            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
    writer.close()  # close tensorboard

def test():
    test_set = DatasetFromHdf5("demo_cave_patches.h5")
    print(torch.cuda.get_device_name(0))
    num_testing = 64
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1)
    sz = 64
    output_HRHSI = np.zeros((num_testing, sz, sz, 31))
    UP_LRHSI = np.zeros((num_testing, sz, sz, 31))
    Highpass= np.zeros((num_testing, sz, sz, 31))
    model = MainNet()
    model = nn.DataParallel(model)
    path_checkpoint = model_folder + "{}.pth".format(1000)
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()
    model = nn.DataParallel(model)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size(), parameters)

    for iteration, batch in enumerate(testing_data_loader, 1):
        _,  LRHSI, HRMSI = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        LRHSI = LRHSI.cuda()
        HRMSI = HRMSI.cuda()
        with torch.no_grad():
            output_HRHSIone,UP_LRHSIone,Highpassone = model(LRHSI, HRMSI)
        output_HRHSI[iteration-1,:,:,:] = output_HRHSIone.permute([0, 2, 3, 1]).cpu().detach().numpy()
        UP_LRHSI[iteration-1,:,:,:] = UP_LRHSIone.permute([0, 2, 3, 1]).cpu().detach().numpy()
        Highpass[iteration - 1, :, :, :] = Highpassone.permute([0, 2, 3, 1]).cpu().detach().numpy()
    sio.savemat('PatchOutput-cave.mat',{'output': output_HRHSI})
###################################################################
# ------------------- Main Function  -------------------
###################################################################
if __name__ == "__main__":
    train_or_not = 0
    test_or_not = 1

    if train_or_not:
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.device_count())
        train_set = DatasetFromHdf5('train_cave.h5')  # creat data for training
        training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
        validate_set = DatasetFromHdf5('validation_cave.h5')  # creat data for validation
        validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
        train(training_data_loader, validate_data_loader)#, start_epoch=200)  # call train function (call: Line 53)

    if test_or_not:
        print("----------------------------testing-------------------------------")
        test()

