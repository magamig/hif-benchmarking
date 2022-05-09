import argparse
import os
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json
from data.load_data import loadingData
from data.load_test_data import loadingTestData
from BlockModule import DeepShare
from basicModule import *
import scipy.io
# loss
from Loss import HybridLoss, CrossEntropy2d
from metrics import quality_assessment
from torch.autograd import Variable

# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = ''


def main():
    # parsers UseUnLabeledMixUp
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False, default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, required=True, help="batch size, default set to 64")
    train_parser.add_argument("--UseLabeledSpectralMixUp", type=int, default=0, help="if we use gan loss, 0  for false, 1 for yes")
    train_parser.add_argument("--theta_LabeledSpectralMixUp", type=int, default=0, help="if we use gan loss, 0  for false, 1 for yes")
    train_parser.add_argument("--UseUnlabelConsistency", type=int, default=0, help="if we use unlabeled consistency, 0  for false, 1 for yes")
    train_parser.add_argument("--UseRGB", type=int, default=0, help="if we use rgb, 0  for false, 1 for yes")
    train_parser.add_argument("--epochs", type=int, default=10,  help="epochs, default set to 20")
    train_parser.add_argument("--conversionMat_path", type=str, default="./data/conversion_8channels.mat", help="path to conversion matrix transforming spectral images of 31 channels to images of 8 channels ")
    train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    train_parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--train_dir_mslabel", type=str, required=True, help="directory of train spectral dataset")
    train_parser.add_argument("--train_dir_msunlabel", type=str,
                              help="directory of train spectral dataset")
    train_parser.add_argument("--eval_dir_ms", type=str, help="directory of evaluation spectral dataset")
    train_parser.add_argument("--test_dir", type=str, required=True, help="directory of test spectral dataset")
    train_parser.add_argument("--train_dir_rgb", type=str, help="directory of train rgb dataset")
    train_parser.add_argument("--theta_rgb", type=int, default=3, help="how many times of rgb images regard to ms images")
    train_parser.add_argument("--theta_unlabel", type=int, default=3, help="how many times of unlabel ms images regard to ms label images")
    train_parser.add_argument("--data_train_num", type=int, required=True, help="how many .mat files used in each epoch")
    train_parser.add_argument("--data_eval_num", type=int, help="how many .mat files used in each epoch")
    train_parser.add_argument("--data_test_num", type=int, required=True, help="how many .mat files used in each epoch")
    train_parser.add_argument("--model_title", type=str, default="DeepShare",
                              help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 7)")


    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False, default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 7)")
    test_parser.add_argument("--test_dir", type=str, required=True, help="directory of test spectral dataset")
    test_parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")
    test_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    test_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    test_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    test_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    test_parser.add_argument("--n_colors", type=int, required=True, help="n_colors, default set to 31")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    test_parser.add_argument("--model_title", type=str, default="DeepShare",
                              help="model_title, default set to model_title")
    test_parser.add_argument("--result_path", type=str, default="./Result",
                             help="result_path, directory of result")
    test_parser.add_argument("--data_test_num", type=int, required=True, help="how many .mat files used in each epoch")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass


bce_loss = torch.nn.BCEWithLogitsLoss()



def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(device)
    criterion = CrossEntropy2d().to(device)

    return criterion(pred, label)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    #load conversion_matrix and by multiplying the conversion matrix with images, we can get images with 8 channels
    conversion_matix = scipy.io.loadmat(args.conversionMat_path) #("~/Data/Chikusei/conversion_matrix_128_8.mat") #
    conversion_matix = np.array(conversion_matix['conversion_matrx'], dtype=np.float32)
    conversion_matix = torch.from_numpy(conversion_matix.copy())
    conversion_matix = conversion_matix.to(device)

    print('===> Loading datasets')

    train_mslabel_set = loadingData(image_dir=args.train_dir_mslabel, augment=True, total_num=args.data_train_num)
    train_msunlabel_set = loadingData(image_dir=args.train_dir_msunlabel, augment=True, total_num=args.theta_unlabel*args.data_train_num)
    train_rgb_set = loadingData(image_dir=args.train_dir_rgb, augment=True, total_num=args.theta_rgb*args.data_train_num)
    eval_mslabel_set = loadingData(image_dir=args.eval_dir_ms, augment=False, total_num=args.data_eval_num)

    eval_ms_loader = DataLoader(eval_mslabel_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    if args.dataset_name == 'Cave':
        colors = 31
    elif args.dataset_name == 'NTIRE2020':
        colors = 31
    elif args.dataset_name == 'Harvard':
        colors = 31
    else:
        colors = 128

    print('===> Building model')
    if args.model_title =="DeepShare":
        net = DeepShare(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=colors, n_blocks=args.n_blocks, n_feats=args.n_feats,
                n_scale=args.n_scale, res_scale=0.1, use_share=args.use_share, conv=default_conv)


    model_title = args.dataset_name + "_" + args.model_title + '_Blocks=' + str(args.n_blocks) + '_Subs' + str(
        args.n_subs) + '_Ovls' + str(args.n_ovls) + '_Feats=' + str(args.n_feats)
    model_name = './checkpoints/' + model_title + "_ckpt_epoch_" + str(40) + ".pth"
    args.model_title = model_title

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
            #state_dict = torch.load(model_name)
            #net.load_state_dict(state_dict, strict=False)

        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()


    # loss functions to choose
    # mse_loss = torch.nn.MSELoss()
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter_mslabel = meter.AverageValueMeter()
    epoch_meter_mslabelmixup = meter.AverageValueMeter()
    epoch_meter_mslabelrgb = meter.AverageValueMeter()
    epoch_meter_msunlabelmixup = meter.AverageValueMeter()
    epoch_meter_msunlabel = meter.AverageValueMeter()
    epoch_meter_rgb = meter.AverageValueMeter()
    writer = SummaryWriter('runs/' + model_title + '_' + str(time.ctime()))



    print('===> Start training')

    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e + 1, args.epochs)


        epoch_meter_mslabel.reset()
        print("Start epoch {}, labeled ms learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        epoch_meter_mslabelmixup.reset()
        epoch_meter_msunlabel.reset()
        epoch_meter_mslabelrgb.reset()
        epoch_meter_msunlabelmixup.reset()
        epoch_meter_rgb.reset()
        iteration = 0
        train_mslabel_loader = DataLoader(train_mslabel_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
        train_msunlabel_loader = DataLoader(train_msunlabel_set, batch_size=args.theta_unlabel*args.batch_size, num_workers=4, shuffle=True)
        train_rgb_loader = DataLoader(train_rgb_set, batch_size=args.theta_rgb * args.batch_size, num_workers=4, shuffle=True)
        train_mslabel_iter = iter(train_mslabel_loader)
        train_msunlabel_iter = iter(train_msunlabel_loader)
        train_rgb_iter = iter(train_rgb_loader)

        for batch_mslabel, batch_msunlabel, batch_rgb in zip(train_mslabel_iter, train_msunlabel_iter, train_rgb_iter):
            # training for spectral images


            x, lms, gt = batch_mslabel
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y_ms_l = net(x, lms, modality="spectral")
            loss = h_loss(y_ms_l, gt)
            epoch_meter_mslabel.add(loss.item())
            loss.backward()
            optimizer.step()


        # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): ms Loss: {:.6f}".format(time.ctime(),
                                                                                            args.n_blocks,
                                                                                            args.n_subs,
                                                                                            args.n_feats,
                                                                                            args.gpus, e + 1,
                                                                                            iteration + 1,
                                                                                            len(train_mslabel_loader),
                                                                                            loss.item()))
                n_iter = e * len(train_mslabel_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss_ms', loss, n_iter)

            if args.UseLabeledSpectralMixUp == 1 and e<=(args.epochs-1):
                # spectral mixup on labeled data

                bsize, fsize, hsize, wsize = gt.shape
                for mixid in range(args.theta_LabeledSpectralMixUp):
                    conversion_matrix_rand = np.random.rand(fsize,fsize)
                    conversion_matrix_rand = np.array(conversion_matrix_rand/(np.mean(np.sum(conversion_matrix_rand, axis=0))), dtype=np.float32)
                    conversion_matrix_rand = torch.from_numpy(conversion_matrix_rand)
                    conversion_matrix_rand = conversion_matrix_rand.to(device)

                    x_mixup, lms_mixup, gt_mixup = (x + conversion(x, conversion_matrix_rand))/2, (lms + conversion(lms,conversion_matrix_rand))/2, (gt + conversion(gt, conversion_matrix_rand))/2
                    optimizer.zero_grad()
                    y_ms_l = net(x_mixup, lms_mixup, modality="spectral")
                    loss = h_loss(y_ms_l, gt_mixup)
                    epoch_meter_mslabelmixup.add(loss.item())
                    loss.backward()
                    optimizer.step()

                    # tensorboard visualization
                    if (iteration + log_interval) % log_interval == 0:
                        print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): ms spectral mixup Loss: {:.6f}".format(time.ctime(),
                                                                                                        args.n_blocks,
                                                                                                        args.n_subs,
                                                                                                        args.n_feats,
                                                                                                        args.gpus, e + 1,
                                                                                                        iteration + 1,
                                                                                                        len(train_mslabel_loader),
                                                                                                        loss.item()))
                        n_iter = e * len(train_mslabel_loader) + iteration + 1
                        writer.add_scalar('scalar/train_loss_ms_mixup', loss, n_iter)



            # training for rgb images batch 1
            if args.UseRGB == 1 and e<=(args.epochs-1):
                x, lms, gt_rgb = batch_rgb
                x, lms, gt_rgb = x.to(device), lms.to(device), gt_rgb.to(device)
                y_rgb = torch.zeros(gt_rgb.size())
                for i in range(0, args.theta_rgb):
                    x_i, lms_i, gt_i = x[i*args.batch_size:(i+1)*args.batch_size,:,:,:], lms[i*args.batch_size:(i+1)*args.batch_size,:,:,:], gt_rgb[i*args.batch_size:(i+1)*args.batch_size,:,:,:]
                    optimizer.zero_grad()
                    y_rgb_i = net(x_i, lms_i, modality="rgb")
                    y_rgb[i*args.batch_size:(i+1)*args.batch_size,:,:,:] = y_rgb_i
                    loss = h_loss(y_rgb_i, gt_i)
                    epoch_meter_rgb.add(loss.item())
                    loss.backward()
                    optimizer.step()

                    if (iteration + log_interval) % log_interval == 0:
                        print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): rgb Loss: {:.6f}".format(time.ctime(),
                                                                                                         args.n_blocks,
                                                                                                         args.n_subs,
                                                                                                         args.n_feats,
                                                                                                         args.gpus,
                                                                                                         e + 1,
                                                                                                         iteration + 1,
                                                                                                         len(train_rgb_loader),
                                                                                                         loss.item()))
                        n_iter = e * len(train_rgb_loader) + iteration + 1
                        writer.add_scalar('scalar/train_loss_rgb', loss, n_iter)

            # #transfer HR 31 channels to 8 channels, then check the consistency.
            if args.UseUnlabelConsistency == 1 and e<=(args.epochs/2):
                x, lms, gt = batch_msunlabel
                x, lms = x.to(device), lms.to(device)
                x_cvt, lms_cvt = conversion(x, conversion_matix), conversion(lms, conversion_matix)
                x_cvt, lms_cvt = x_cvt.to(device), lms_cvt.to(device)

                for j in range(0, args.theta_unlabel):
                    x_j, lms_j, x_cvt_j, lms_cvt_j = x[j*args.batch_size:(j+1)*args.batch_size,:,:,:], lms[j*args.batch_size:(j+1)*args.batch_size,:,:,:], x_cvt[j*args.batch_size:(j+1)*args.batch_size,:,:,:], lms_cvt[j*args.batch_size:(j+1)*args.batch_size,:,:,:]
                    optimizer.zero_grad()
                    y_ms = net(x_j, lms_j, modality="spectral")
                    rgb_from_ms = conversion(y_ms, conversion_matix)
                    y_rgb_cvt = net(x_cvt_j, lms_cvt_j, modality="rgb")
                    loss = h_loss(rgb_from_ms, y_rgb_cvt)

                    epoch_meter_msunlabel.add(loss.item())

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # tensorboard visualization
                    if (iteration + log_interval) % log_interval == 0:
                        print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): unlabeled ms consistency Loss: {:.6f}".format(
                            time.ctime(),
                            args.n_blocks,
                            args.n_subs,
                            args.n_feats,
                            args.gpus, e + 1,
                                       iteration + 1,
                            len(train_mslabel_loader),
                            loss.item()))
                        n_iter = e * len(train_mslabel_loader) + iteration + 1
                        writer.add_scalar('scalar/train_loss_unlabeled_ms_mixup', loss, n_iter)


            iteration += 1

        print("===> {}\tEpoch {} Training mslabel Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                              epoch_meter_mslabel.value()[0]))
        print("===> {}\tEpoch {} Training msunlabel Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                                 epoch_meter_msunlabel.value()[0]))
        print("===> {}\tEpoch {} Training rgb Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                              epoch_meter_rgb.value()[0]))

        # run validation set every epoch
        eval_loss_ms = validate(args, eval_ms_loader, "spectral", net, L1_loss, args.theta_rgb)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss_mslabel', epoch_meter_mslabel.value()[0], iteration)
        writer.add_scalar('scalar/avg_validation_loss_ms', eval_loss_ms, iteration)
        writer.add_scalar('scalar/avg_epoch_loss_msunlabel', epoch_meter_msunlabel.value()[0], iteration)
        writer.add_scalar('scalar/avg_epoch_loss_rgb', epoch_meter_rgb.value()[0], iteration)
        # save model weights at checkpoints every 10 epochs
        #if (e + 1) % 5 == 0:
        save_checkpoint(args, net, e + 1)

    # save model after training
    net.eval().cpu()
    save_model_filename = model_title + "_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(args.save_dir, save_model_filename)
    if torch.cuda.device_count() > 1:
        torch.save(net.module.state_dict(), save_model_path)
    else:
        torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


    ## Save the testing results
    print("Running testset")
    print('===> Loading testset')
    test_set = loadingTestData(image_dir=args.test_dir, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net.eval().cuda()
    with torch.no_grad():
        output = []
        test_number = 0
        for i, (x, lms, gt) in enumerate(test_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            y = net(x, lms, modality="spectral")
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    save_dir = model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = model_title + '_' + str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))



def conversion(m_input, m_conversion):
    b, c, h, w = m_input.shape
    m_input = m_input.permute(0, 3, 2, 1)
    m_input = torch.reshape(m_input, (b * w * h, c))
    x, y = m_conversion.shape
    if c == x:
        m_new = torch.matmul(m_input, m_conversion)
        m_new = torch.reshape(m_new, (b, w, h, y))
        m_new = m_new.permute(0, 3, 2, 1)

        return m_new
    else:
        raise
    #    print("Wrong dimensions for matrix multiplication")


def sum_dict(a, b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch, total_epoch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    #lr = start_lr * (0.1 ** (epoch // 30))
    lr = start_lr * (0.3 ** (epoch // 5))
    if epoch==total_epoch_num:
        lr = lr * 0.3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_D(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    #lr = start_lr * (0.1 ** (epoch // 30))
    lr = start_lr * (0.3 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(args, loader, modality, model, criterion, theta):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        if modality == "spectral":
            for i, (ms, lms, gt) in enumerate(loader):
                ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
                # y = model(ms)
                y = model(ms, lms, modality)
                loss = criterion(y, gt)
                epoch_meter.add(loss.item())
        elif modality == "rgb":
            for i, (ms, lms, gt) in enumerate(loader):
                for j in range(0, theta):
                    ms_j, lms_j, gt_j = ms[j*args.batch_size:(j+1)*args.batch_size,:,:,:], lms[j*args.batch_size:(j+1)*args.batch_size,:,:,:], gt[j*args.batch_size:(j+1)*args.batch_size,:,:,:]
                    ms_j, lms_j, gt_j = ms_j.to(device), lms_j.to(device), gt_j.to(device)
                    # y = model(ms)
                    y = model(ms_j, lms_j, modality)
                    loss = criterion(y, gt_j)
                    epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch {} evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), modality, epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    test_set = loadingTestData(image_dir=args.test_dir, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        model = DeepShare(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=args.n_colors, n_blocks=args.n_blocks, n_feats=args.n_feats,
                      n_scale=args.n_scale, res_scale=0.1, use_share=True, conv=default_conv)

        state_dict = torch.load(args.model_dir)
        model.load_state_dict(state_dict, strict=False)
        #checkpoint = torch.load(args.model_dir)
        #model.load_state_dict(checkpoint["model"].state_dict())

        model.to(device).eval()
        mse_loss = torch.nn.MSELoss()
        output = []
        test_number = 0
        for i, (x, lms, gt) in enumerate(test_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            y = model(x, lms, modality="spectral")

            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    save_dir = args.result_path + args.model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = args.model_title + '_' + str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))


def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))



if __name__ == "__main__":
    main()
