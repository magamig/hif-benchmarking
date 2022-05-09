import torch.utils.data as data
import torch
import glob
import numpy as np
import torchvision.transforms as transforms
import tifffile as tiff
import scipy.misc as smi
import cv2
import time
from PIL import Image
import scipy.io as scio

def get_lrhsi(img, degradation_mode): 
    # print("degradation_mode:", degradation_mode)
    if degradation_mode == 0:
        scale_factor = 8
        dim = np.shape(img)
        img_down = np.zeros([dim[0], int(dim[1]/scale_factor), int(dim[2]/scale_factor)])
        img_rebuild = np.zeros(dim)
        # gaussian blur kernel, size:8x8, standrad deviation:3
        kernel = np.array([[0.0067, 0.0094, 0.0118, 0.0131, 0.0131, 0.0118, 0.0094, 0.0067],
              [0.0094, 0.0131, 0.0164, 0.0183, 0.0183, 0.0164, 0.0131, 0.0094],
              [0.0118, 0.0164, 0.0205, 0.0229, 0.0229, 0.0205, 0.0164, 0.0118],
              [0.0131, 0.0183, 0.0229, 0.0256, 0.0256, 0.0229, 0.0183, 0.0131],
              [0.0131, 0.0183, 0.0229, 0.0256, 0.0256, 0.0229, 0.0183, 0.0131],
              [0.0118, 0.0164, 0.0205, 0.0229, 0.0229, 0.0205, 0.0164, 0.0118],
              [0.0094, 0.0131, 0.0164, 0.0183, 0.0183, 0.0164, 0.0131, 0.0094],
              [0.0067, 0.0094, 0.0118, 0.0131, 0.0131, 0.0118, 0.0094, 0.0067]]) 
        kernel = [kernel] * dim[0]
        for i in range(int(dim[1]/scale_factor)):
            for j in range(int(dim[2]/scale_factor)):
                img_down[:, i, j] = np.sum(np.sum(img[:, i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] * kernel, axis=1), axis=1)
    elif degradation_mode == 1:
        scale_factor = 8
        dim = np.shape(img)
        img_down = np.zeros([dim[0], int(dim[1]/scale_factor), int(dim[2]/scale_factor)])
        img_rebuild = np.zeros(dim)
        # uniform blur kernel, size: scale_factor * scale_factor
        kernel = np.ones([int(scale_factor), int(scale_factor)]) / (scale_factor**2)
        kernel = [kernel] * dim[0]
        for i in range(int(dim[1]/scale_factor)):
            for j in range(int(dim[2]/scale_factor)):
                img_down[:, i, j] = np.sum(np.sum(img[:, i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] * kernel, axis=1), axis=1)
    elif degradation_mode == 2:
        scale_factor = 16
        dim = np.shape(img)
        img_down = np.zeros([dim[0], int(dim[1]/scale_factor), int(dim[2]/scale_factor)])
        img_rebuild = np.zeros(dim)
        # uniform blur kernel, size: scale_factor * scale_factor
        kernel = np.ones([int(scale_factor), int(scale_factor)]) / (scale_factor**2)
        kernel = [kernel] * dim[0]
        for i in range(int(dim[1]/scale_factor)):
            for j in range(int(dim[2]/scale_factor)):
                img_down[:, i, j] = np.sum(np.sum(img[:, i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] * kernel, axis=1), axis=1)
    else:
        scale_factor = 32
        dim = np.shape(img)
        img_down = np.zeros([dim[0], int(dim[1]/scale_factor), int(dim[2]/scale_factor)])
        img_rebuild = np.zeros(dim)
        # uniform blur kernel, size: scale_factor * scale_factor
        kernel = np.ones([int(scale_factor), int(scale_factor)]) / (scale_factor**2)
        kernel = [kernel] * dim[0]
        for i in range(int(dim[1]/scale_factor)):
            for j in range(int(dim[2]/scale_factor)):
                img_down[:, i, j] = np.sum(np.sum(img[:, i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] * kernel, axis=1), axis=1)
    
    for i in range(dim[0]):
        img_down_slice = Image.fromarray(img_down[i, :, :])
        img_rebuild[i, :, :] = img_down_slice.resize((dim[1],dim[2]), Image.BICUBIC)
        # img_rebuild[i, :, :] = cv2.resize(img_down[i, :, :],(dim[1],dim[2]),interpolation=cv2.INTER_LINEAR)
    return img_rebuild.astype(np.float32)

def flip(img, random_flip):
    # input:3D array to flip
    # output:3D array flipped
    channel = np.shape(img)[0]
    # Random horizontal flipping
    if random_flip[0] > 0.5:
        for i in range(channel):
            img[i, :, :] = np.fliplr(img[i, :, :])
    # Random vertical flipping
    if random_flip[1] > 0.5:
        for i in range(channel):
            img[i, :, :] = np.flipud(img[i, :, :])
    # Rotate img by 90 degrees
    if random_flip[2] > 0.5:
        for i in range(channel):
            img[i, :, :] = np.rot90(img[i, :, :])
    return img

class Dataset_cave_train(data.Dataset):
	def __init__(self, file_train_path):
		self.tif_list = glob.glob(file_train_path+'/*.tif')
		self.transforms = transforms.ToTensor() 
		# self.scale_factor = scale_factor

	def __getitem__(self, index):
		index = index % len(self.tif_list)
		img = tiff.imread(self.tif_list[index])
		# print(int(time.time()*1000))
		np.random.seed(int(time.time()*1000)%1000000)
		topleft = np.random.random_integers(0, 512-128, 2)
		degradation_mode = np.random.random_integers(0, 4)

		img = img[:, topleft[0]:topleft[0]+128, topleft[1]:topleft[1]+128]

		random_flip = np.random.random(3)
		img = flip(img, random_flip)

		img_hsi = img[0:31, :, :]
		img_rgb = img[31:31+3, :, :]
		img_hsi_rebuild = get_lrhsi(img_hsi, degradation_mode)

		data_hsi = self.transforms(img_hsi_rebuild.transpose([1, 2, 0])) 
		data_rgb = self.transforms(img_rgb.transpose([1, 2, 0]))
		label = self.transforms(img_hsi.transpose([1, 2, 0]))
		
		return data_hsi, data_rgb, label

	def __len__(self):
		return 100 * len(self.tif_list) # num of patchs is 100 in one raw train image

class Dataset_cave_val(data.Dataset):
	def __init__(self, file_test_path):
		self.tif_list = glob.glob(file_test_path+'/*.tif')
		self.transforms = transforms.ToTensor() 
		# self.scale_factor = scale_factor

	def __getitem__(self, index):
		index = index % len(self.tif_list)
		img = tiff.imread(self.tif_list[index])

		np.random.seed(int(time.time()*1000)%1000000)
		topleft = np.random.random_integers(0, 512-128, 2)
		degradation_mode = np.random.random_integers(0, 4)
		img = img[:, topleft[0]:topleft[0]+128, topleft[1]:topleft[1]+128]

		img_hsi = img[0:31, :, :]
		img_rgb = img[31:31+3, :, :]
		img_hsi_rebuild = get_lrhsi(img_hsi, degradation_mode)
		# img_rgb = np.random.random_integers(256, size=[3, 64, 64]).astype(np.uint8)

		data_hsi = self.transforms(img_hsi_rebuild.transpose([1, 2, 0])) 
		data_rgb = self.transforms(img_rgb.transpose([1, 2, 0]))
		label = self.transforms(img_hsi.transpose([1, 2, 0]))
		
		return data_hsi, data_rgb, label

	def __len__(self):
		return 5 * len(self.tif_list)

class Dataset_cave_test(data.Dataset):
	def __init__(self, file_test_path, degradation_mode):
		self.tif_list = glob.glob(file_test_path+'/*.tif')
		self.transforms = transforms.ToTensor() 
		self.degradation_mode = degradation_mode

	def __getitem__(self, index):
		img = tiff.imread(self.tif_list[index])

		img_hsi = img[0:31, :, :]
		img_rgb = img[31:31+3, :, :]
		img_hsi_rebuild = get_lrhsi(img_hsi, self.degradation_mode)
		# img_rgb = np.random.random_integers(256, size=[3, 512, 512]).astype(np.uint8)

		data_hsi = self.transforms(img_hsi_rebuild.transpose([1, 2, 0])) 
		data_rgb = self.transforms(img_rgb.transpose([1, 2, 0]))
		label = self.transforms(img_hsi.transpose([1, 2, 0]))
		
		return data_hsi, data_rgb, label

	def __len__(self):
		return len(self.tif_list)

# # train
# file = './data/train/'
# a = Dataset_cave_train(file)
# hsi, rgb, lb = a.__getitem__(313)

# # # test
# # file = './data/test/'
# # a = Dataset_cave_test(file)
# # hsi, rgb, lb = a.__getitem__(1)

# print(lb.dtype)

# print(hsi.size())
# hsi = hsi.numpy()*255
# tiff.imsave('./data/hsi.tif', hsi.astype(np.uint8))

# lb = lb.numpy()*255
# tiff.imsave('./data/lb.tif', lb.astype(np.uint8))

# rgb = rgb.numpy()*255
# tiff.imsave('./data/rgb.tif', rgb.astype(np.uint8))
