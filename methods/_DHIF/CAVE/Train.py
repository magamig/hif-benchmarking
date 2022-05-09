from Model import HSI_Fusion
from CAVE_Dataset import cave_dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__=="__main__":

    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='./Data/Train/', type=str,
                        help='Path of the training data')
    parser.add_argument("--sizeI", default=96, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=4, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser.add_argument("--seed", default=1, type=int, help='Random seed')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    opt = parser.parse_args()

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print(opt)

    ## New model
    print("===> New Model")
    model = HSI_Fusion(Ch=31, stages=4, sf=opt.sf)

    ## set the number of parallel GPUs
    print("===> Setting GPU")
    model = dataparallel(model, 1)

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    ## Load training data
    key = 'Train.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 20)

    ## Load trained model
    initial_epoch = findLastCheckpoint(save_dir="./Checkpoint/f8/Model")
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join("./Checkpoint/f8/Model", 'model_%03d.pth' % initial_epoch))

    ## Loss function
    criterion = nn.L1Loss()

    ## optimizer and scheduler
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95)

    ## pipline of training
    for epoch in range(initial_epoch, 500):
        model.train()

        dataset = cave_dataset(opt, HR_HSI, HR_MSI)
        loader_train = tud.DataLoader(dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        start_time = time.time()
        for i, (LR, RGB, HR) in enumerate(loader_train):
            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            out = model(RGB.cuda(), LR.cuda())

            loss = criterion(out, HR.cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 2000 == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(dataset)// opt.batch_size, epoch_loss / ((i+1) * opt.batch_size), datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(dataset), elapsed_time))
        torch.save(model, os.path.join("./Checkpoint/f8/Model", 'model_%03d.pth' % (epoch + 1)))  # save model
