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
import lmdb
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)
class Hyper_dataset(Dataset):
    """
    
    get the Hyperspectral image and corrssponding RGB image  
    use all data : high resolution HSI, high resolution MSI, low resolution HSI
    """
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True):
        self.read_data = use_generated_data
        self.direct_data = use_all_data
        self.data_name = data_name
        assert(output_shape in [32,64,128,256])
        assert(Training_mode in ['Train','Test'])
        assert(self.data_name in ['CAVE','ICVL'])
        self.TM = Training_mode
        self.output_shape = output_shape  
        # if self.direct_data == True:
        if self.data_name == 'CAVE':
            self.hsi_data = np.load("/home/zhu_19/Hyperspectral_image/data/new_hsi_data.npy")
            self.msi_data = np.load("/home/zhu_19/Hyperspectral_image/data/new_msi_data.npy")
            self.num_pre_img = self.hsi_data.shape[-1]//self.output_shape 
            self.len = self.hsi_data.shape[0] * self.num_pre_img**2 
            print(self.len)
            # self.train_len = round(self.len*0.645)
            # self.test_len = round(self.len*0.375)//16
            self.train_len = round(self.len*0.645)
            self.test_len = round(self.len*0.375)//16
            self.sponse = scio.loadmat("/home/zhu_19/Hyperspectral_image/co_0/response coefficient.mat")
            self.rep = self.sponse['iniA'].transpose(1,0)
            self.max_min = np.load("/home/zhu_19/Hyperspectral_image/co_0/max_min.npy")
            self.max = np.max(self.max_min)
            self.min = np.min(self.max_min)
            # self.shuffle_index = [2,31,25,6,27,15,19,14,12,28,26,29,8,13,22,7,24,30,10,23,18,17,21,3,9,4,20,5,16,11,1]
        elif self.data_name == 'ICVL':
            # self.DName = []
            # self.path = '/home/zhu_19/Hyperspectral_image/data/ICVL/'
            self.path = '/home/zhu_19/data/lmdb/data/'
            self.env = lmdb.open(self.path)
            self.txn = self.env.begin()
            self.name = get_name()
            self.num_pre_img = 1024 // self.output_shape
            self.len = len(self.name)*self.num_pre_img*self.num_pre_img
            self.train_len = round(len(self.name)*0.7)*self.num_pre_img*self.num_pre_img
            self.test_len = self.len-self.train_len
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
        return np.concatenate([self.zoom_img_(img,output_shape = output_shape)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img_(self,input_img,output_shape):
        return input_img.reshape(input_img.shape[0],output_shape,-1).mean(-1).swapaxes(0,1).reshape(output_shape,output_shape,-1).mean(-1).swapaxes(0,1)
    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1,2,0),dsize=(self.shape1,self.shape1)),dsize = (self.output_shape , self.output_shape)).transpose(2,0,1)
    def __getitem__(self, index):
        if self.TM == 'Test':
            index = index + self.train_len//16
        # if self.direct_data == True:
        index_img = index // self.num_pre_img**2 
        # index_img = self.shuffle_index[index_img]-1
        index_inside_image = index % self.num_pre_img**2 
        index_row = index_inside_image // self.num_pre_img 
        index_col = index_inside_image % self.num_pre_img
        if self.data_name == 'CAVE':
            if self.TM == 'Train':
                hsi_g = self.hsi_data[index_img,:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]
                # msi = self.msi_data[index_img,:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]
                msi = np.tensordot(hsi_g,self.rep,(0,0)).transpose((2,0,1))/(2**16-1)
                hsi_g = hsi_g.astype(np.float)/(2**16-1)
                # msi = msi.astype(np.float)/(2**8-1)
            elif self.TM == 'Test':
                # index = self.shuffle_index[index]-1
                hsi_g = self.hsi_data[index,:,:,:]
                msi = np.tensordot(hsi_g,self.rep,(0,0)).transpose((2,0,1))/(2**16-1)
                hsi_g = hsi_g.astype(np.float)/(2**16-1)
        elif self.data_name == 'ICVL':
            index_name = index // self.num_pre_img**2
            temp_name = self.name[index_name]
            a,_ = temp_name.split('.')
            nhsi = a+'_hsi'
            nrgb = a+'_rgb'
            hsi_g = self.txn.get(nhsi.encode())
            hsi_g = np.frombuffer(hsi_g)
            hsi_g = np.reshape(hsi_g,(31,1024,1024))
            rgb = self.txn.get(nrgb.encode())
            rgb = np.frombuffer(rgb)
            rgb = np.reshape(rgb,(3,1024,1024))

            hsi_g = hsi_g[:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]/4000
            msi = rgb[:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]/255
        hsi = self.zoom_img(hsi_g,1/32)
        hsi_resize = hsi
        hsi_8 = self.zoom_img(hsi_g, 1/8)
        hsi_2 = self.zoom_img(hsi_g, 1/2)
        msi_8 = self.zoom_img(msi,1/8)
        msi_2 = self.zoom_img(msi,1/2)
        return ((hsi,hsi_8,hsi_2), hsi_g, hsi_resize, (msi_8,msi_2,msi))
        