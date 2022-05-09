# coding=UTF-8
# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image,ImageOps
import numbers
import random
from skimage.color import rgb2ycbcr
import matplotlib.pyplot as plt
from util import rgb2ycbcr_tensor
import numpy as np
from scipy.io import loadmat
from clean_util import H_z


IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def crop(img, i, j, h, w):
    """Crop the given PIL Image."""
    return img.crop((j, i, j + w, i + h))


def augment(HR, LR , RGB, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, factor=0,padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.factor = factor

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        h, w , _= img.shape
        # w, h = img.shape

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)


        return i, j,  th, tw

    # def __call__(self, img1, img8):
    def __call__(self, clean, noisy):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        h, w , _ = clean.shape
        # W, H , _ = HR.size


        if self.padding is not None:
            img1 = F.pad(img1, self.padding, self.fill, self.padding_mode)
            img2 = F.pad(img2, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img1.size[0] < self.size[1]:
            img1 = F.pad(img1, (self.size[1] - img1.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img2.size[0] < self.size[1]:
            img2 = F.pad(img2, (self.size[1] - img2.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img1.size[1] < self.size[0]:
            img1 = F.pad(img1, (0, self.size[0] - img1.size[1]), self.fill, self.padding_mode)
        if self.pad_if_needed and img2.size[1] < self.size[0]:
            img2 = F.pad(img2, (0, self.size[0] - img2.size[1]), self.fill, self.padding_mode)

        i,  j, h, w = self.get_params(clean, self.size)

        # return crop(LR, i, j, h, w), crop(HR, i1, j1, h * self.factor, w * self.factor),crop(RGB, i1, j1, h * self.factor, w * self.factor)
        return (clean[i : i + h,  j : j+w, :], noisy[i : i + h ,  j : j+w, :])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self,fft_B, patch_size,trainlist, HR_dir='',RGB_dir='', task='', channel=3,factor = 0, crop=True,
                 loader=default_loader):  # , root, fn, crop=True, loader=default_loader):

        self.fft_B = fft_B
        self.data_dir = HR_dir
        self.noisy_dir = RGB_dir
        self.task = task
        self.pathlist = self.loadpath(trainlist)
        self.HR_list = []
        self.RGB_list = []
        for i in range(len(self.pathlist)):
            im_path = self.pathlist[i]
            clean = loadmat(im_path)
            clean = clean['hsi']
            # if "img" in im_path:
            #     clean = 16.0 * clean
            # else:
            #     clean = clean / 65535.0
            self.HR_list.append(clean)
            p = im_path.split('/')[-1]
            noisy = loadmat(self.noisy_dir + '/' + p)
            noisy = noisy['rgb']
            # noisy = noisy / 255.0
            self.RGB_list.append(noisy)

        self.count = len(self.pathlist)
        self.channel = channel
        self.crop = crop
        self.factor = factor
        self.patch_size = patch_size
        if crop:
            self.crop_LR = RandomCrop(patch_size)

        else:
            self.crop_LR = False

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std),
            ])
        self.loader = loader

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

    def Para_setting(self,kernel_type,sf,sz):

        if kernel_type == 'uniform_blur':
            psf = np.ones([sf, sf]) / (sf * sf)
        # elif kernel_type == 'Gaussian':
        #     psf = fspecial('gaussian', 8, 3);
        fft_B = pypher.psf2otf(psf, sz)
        fft_BT = np.conj(fft_B)
        return fft_B, fft_BT



    def __getitem__(self, index):

        HR = self.HR_list[index]
        RGB = self.RGB_list[index]

        if self.crop:
            HR, RGB  = self.crop_LR(HR, RGB)
            # HR,  LR, RGB  = augment(HR, LR, RGB)
        shape = HR.shape

        #-----------------------------------------------------
        #  generate LR with fft and ifft
        # HR = torch.FloatTensor(HR).permute(2, 0, 1)
        HR = np.transpose(HR,[2,0,1])
        RGB = np.transpose(RGB,[2,0,1])
        HR = torch.FloatTensor(HR)
        RGB = torch.FloatTensor(RGB)

        LR = H_z(HR, self.factor, self.fft_B )

        #-----------------------------------------------------


        # LR = np.zeros((int(shape[0]/self.factor), int(shape[1]/self.factor), shape[2]))
        # for j in range(0,self.factor):
        #     for k in range(0,self.factor):
        #         LR = LR + HR[ j:512:self.factor, k:512:self.factor, :] / self.factor / self.factor
        ########################################################################

        #######################################################################

        return (HR, LR, RGB) # LR [1,40,40]  ;GT [1,40,40]

    def __len__(self):
        return self.count



