import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import scipy.ndimage as scin
from scipy import ndimage
from get_name import get_name
import scipy.io as scio
import h5py
# import lmdb
import os
import random
import h5py
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)
class Hyper_dataset(Dataset):
    """
    
    get the Hyperspectral image and corrssponding RGB image  
    use all data : high resolution HSI, high resolution MSI, low resolution HSI
    """
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True):
        # self.path = '/home/zhu_19/Hyperspectral_image/Hyperspectral_image_comparing_method/MHF-net-master/CAVEdata/'
        self.data_name = data_name
        if data_name == 'CAVE':
            self.path = '/home/zhu_19/data/hyperspectral/12_31/CAVEdata_1931/'
            # file_name = os.walk(self.path+'X/')
            # file_name = [i for i in file_name]
            # self.file_name = file_name[0][2]
            # self.file_name = np.load('/home/grads/zhiyuzhu2/hyperspectral_image/hyperspectral/file_name7048.npy')
            name = scio.loadmat('/home/zhu_19/data/instance/file_name.mat')
            self.train_name = name['train']
            self.test_name = name['test']
            self.num_pre_img = 4
            self.train_len = 20*16
            self.test_len = 12
        elif data_name == 'HARVARD':
            self.train_path = '/public/SSD/Harvard/train/'
            file_name = os.walk(self.train_path)
            file_name = [i for i in file_name]
            self.train_name = file_name[0][2]
            self.test_path = '/public/SSD/Harvard/test/'
            file_name = os.walk(self.test_path)
            file_name = [i for i in file_name]
            self.test_name = file_name[0][2]
            self.num_width = int(1040/128)
            self.num_hight = int(1390/128)
            self.train_len = self.num_hight * self.num_width *30
            self.test_len = 20
            self.LR_path = '/public/SSD/Harvard/LR/'
            # self.file = 
            # self.
        self.reps = scio.loadmat('/home/zhu_19/data/instance/resp.mat')['resp']
        self.reps = np.transpose(self.reps,(1,0))
        # self.shuffle_index  = [2,31,25,6,27,15,19,14,12,28,26,29,8,13,22,7,24,30,10,23,18,17,21,3,9,4,20,5,16,32,11,1]
        # save_name = []
        # for i in range(32):
        #     save_name.append(self.file_name[self.shuffle_index[i]-1])
        # scio.savemat('save_name7048.mat',{'dict':save_name})
        self.TM = Training_mode
    def __len__(self):
        if self.TM == 'Train':
            return self.train_len
        elif self.TM == 'Test':
            return self.test_len
    # def zoom_img(self,input_img,ratio_):
    #     return np.concatenate([ndimage.zoom(img,zoom = ratio_)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img(self,input_img,ratio_):
        # return np.concatenate([ndimage.zoom(img,zoom = ratio_)[np.newaxis,:,:] for img in input_img],0)
        output_shape = int(input_img.shape[-1]*ratio_)
        # print(output_shape,'--------------------------------')
        # input_img = cv2.GaussianBlur(input_img,(7,7),2)
        # a = int(1/ratio_)
        # temp = int(a/2)
        # input_img = input_img[:,temp::a,temp::a]
        # 
        return np.concatenate([self.zoom_img_(img,output_shape = output_shape)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img_(self,input_img,output_shape):
        return input_img.reshape(input_img.shape[0],output_shape,-1).mean(-1).swapaxes(0,1).reshape(output_shape,output_shape,-1).mean(-1).swapaxes(0,1)
    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1,2,0),dsize=(self.shape1,self.shape1)),dsize = (self.output_shape , self.output_shape)).transpose(2,0,1)
    def __getitem__(self, index):
        if self.data_name == 'CAVE':
            # if self.TM == 'Test':
            #     index = index + self.train_len//(self.num_pre_img**2)
            if self.TM=='Train':
                # if self.direct_data == True:
                index_img = index // self.num_pre_img**2 
                # index_img = self.shuffle_index[index_img]-1
                index_inside_image = index % self.num_pre_img**2 
                index_row = index_inside_image // self.num_pre_img 
                index_col = index_inside_image % self.num_pre_img
                hsi_g = scio.loadmat(self.path+'X/'+str.rstrip(self.train_name[index_img]))
                # msi = scio.loadmat(self.path+'Y/'+self.file_name[index_img])
                # hsi = scio.loadmat(self.path+'Z/'+self.file_name[index_img])
                temp = hsi_g['msi']
                temp_a = cv2.GaussianBlur(temp,(7,7),2)[3::8,3::8,:]
                hsi_g = hsi_g['msi'][index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                hsi = temp_a[index_row*16:(index_row+1)*16,index_col*16:(index_col+1)*16,:]
                # hsi = hsi['Zmsi'][index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
                # msi = msi['RGB'][index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                msi = np.tensordot(hsi_g,self.reps,(-1,0))
                
                rotTimes = random.randint(0, 3)
                vFlip = random.randint(0, 1)
                hFlip = random.randint(0, 1)
                
                # Random rotation
                for j in range(rotTimes):
                    hsi_g = np.rot90(hsi_g)
                    hsi = np.rot90(hsi)
                    msi = np.rot90(msi)

                # Random vertical Flip   
                for j in range(vFlip):
                    hsi_g = np.flip(hsi_g,axis=1)
                    hsi = np.flip(hsi,axis=1)
                    msi = np.flip(msi,axis=1)
                    # hsi_g = hsi_g[:,::-1,:]
                    # hsi = hsi[:,::-1,:]
                    # msi = msi[:,::-1,:]
            
                # Random Horizontal Flip
                for j in range(hFlip):
                    hsi_g = np.flip(hsi_g,axis=0)
                    hsi = np.flip(hsi,axis=0)
                    msi = np.flip(msi,axis=0)
                    # hsi_g = hsi_g[::-1,:,:]
                    # hsi = hsi[::-1,:,:]
                    # msi = msi[::-1,:,:]
                hsi = np.transpose(hsi,(2,0,1)).copy()
                msi = np.transpose(msi,(2,0,1)).copy()
                hsi_g = np.transpose(hsi_g,(2,0,1)).copy()
                # print('shape of tensor {} {} {}'.format(hsi.shape,msi.shape,hsi_g.shape))
            elif self.TM=='Test':
                hsi_g = scio.loadmat(self.path+'X/'+str.rstrip(self.test_name[index]))
                # msi = scio.loadmat(self.path+'Y/'+self.file_name[index])
                # hsi = scio.loadmat(self.path+'Z/'+self.file_name[index])
                hsi_g = hsi_g['msi']
                hsi = cv2.GaussianBlur(hsi_g,(7,7),2)[3::8,3::8,:]
                msi = np.tensordot(hsi_g,self.reps,(-1,0))
                msi = np.transpose(msi,(2,0,1))
                # hsi_g = hsi_g[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                hsi_g = np.transpose(hsi_g,(2,0,1))
                # hsi = hsi[index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
                hsi = np.transpose(hsi,(2,0,1))
                # hsi = np.transpose(hsi['Zmsi'],(2,0,1))
                # msi = msi[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                # msi = np.transpose(msi['RGB'],(2,0,1))
        elif self.data_name == 'HARVARD':
            index_img = index // (self.num_width*self.num_hight)
            # index_img = self.shuffle_index[index_img]-1
            index_inside_image = index % (self.num_hight*self.num_width)
            index_row = index_inside_image // self.num_hight
            index_col = index_inside_image % self.num_hight
            file=h5py.File('/public/SSD/Harvard/data.h5','r')
            file2=h5py.File('/public/SSD/Harvard/Lr_data.h5','r')
            if self.TM=='Train':
                hsi_g = file[self.train_name[index_img]][:]
                hsi = file2[self.train_name[index_img]][:]
                # hsi_g = scio.loadmat(self.train_path+self.train_name[index_img])['ref']
                # hsi = scio.loadmat(self.LR_path+self.train_name[index_img])['ref']
                # msi = scio.loadmat(self.path+'Y/'+self.file_name[index_img])
                # temp = hsi_g['ref']
                # print('Shape: ------------------ shape of hsi_g:{}'.format(hsi_g['ref'].shape))
                # temp_a = cv2.GaussianBlur(temp,(7,7),2)[3::8,3::8,:]
                # print('Shape: ------------------ shape of read:{} hsi_g:{} index:row{},index:col{}'.format(temp.shape,hsi_g['ref'].shape,index_row,index_col))
                hsi_g = hsi_g[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                hsi = hsi[index_row*16:(index_row+1)*16,index_col*16:(index_col+1)*16,:]
                # print('Shape: +++++++++++++++++++++ shape of read:{} hsi_g:{} index:row{},index:col{}'.format(temp.shape,hsi_g.shape,index_row,index_col))
                # hsi = hsi['Zmsi'][index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
                # msi = msi['RGB'][index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                msi = np.tensordot(hsi_g,self.reps,(-1,0))
                rotTimes = random.randint(0, 3)
                vFlip = random.randint(0, 1)
                hFlip = random.randint(0, 1)
                
                # Random rotation
                for j in range(rotTimes):
                    hsi_g = np.rot90(hsi_g)
                    hsi = np.rot90(hsi)
                    msi = np.rot90(msi)

                # Random vertical Flip   
                for j in range(vFlip):
                    hsi_g = np.flip(hsi_g,axis=1)
                    hsi = np.flip(hsi,axis=1)
                    msi = np.flip(msi,axis=1)
                    # hsi_g = hsi_g[:,::-1,:]
                    # hsi = hsi[:,::-1,:]
                    # msi = msi[:,::-1,:]
            
                # Random Horizontal Flip
                for j in range(hFlip):
                    hsi_g = np.flip(hsi_g,axis=0)
                    hsi = np.flip(hsi,axis=0)
                    msi = np.flip(msi,axis=0)
                    # hsi_g = hsi_g[::-1,:,:]
                    # hsi = hsi[::-1,:,:]
                    # msi = msi[::-1,:,:]
                hsi = np.transpose(hsi,(2,0,1)).copy()
                msi = np.transpose(msi,(2,0,1)).copy()
                hsi_g = np.transpose(hsi_g,(2,0,1)).copy()
                # print('shape of tensor {} {} {}'.format(hsi.shape,msi.shape,hsi_g.shape))
            elif self.TM=='Test':
                hsi_g = file[self.test_name[index_img]][:]
                hsi = file2[self.test_name[index_img]][:]
                # hsi_g = scio.loadmat(self.test_path+self.test_name[index])['ref']
                # hsi = scio.loadmat(self.LR_path+self.test_name[index_img])['ref']
                # msi = scio.loadmat(self.path+'Y/'+self.file_name[index])
                # hsi = scio.loadmat(self.path+'Z/'+self.file_name[index])
                # hsi_g = hsi_g
                # hsi = cv2.GaussianBlur(hsi_g,(7,7),2)[3::8,3::8,:]
                msi = np.tensordot(hsi_g,self.reps,(-1,0))
                msi = np.transpose(msi,(2,0,1))
                # hsi_g = hsi_g[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                hsi_g = np.transpose(hsi_g,(2,0,1))
                # hsi = hsi[index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
                hsi = np.transpose(hsi,(2,0,1))
                # hsi = np.transpose(hsi['Zmsi'],(2,0,1))
                # msi = msi[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
                # msi = np.transpose(msi['RGB'],(2,0,1))


        # hsi = self.zoom_img(hsi_g,1/8)
        hsi_resize = hsi
        # hsi_8 = self.zoom_img(hsi_g, 1/4)
        # hsi_2 = self.zoom_img(hsi_g, 1/2)
        # msi_8 = self.zoom_img(msi,1/4)
        # msi_2 = self.zoom_img(msi,1/2)
        return ((hsi,hsi,hsi), hsi_g, hsi_resize, (msi,msi,msi))
        