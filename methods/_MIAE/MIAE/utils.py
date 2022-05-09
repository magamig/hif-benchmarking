# Author: JianJun Liu
# Date: 2022-1-13
import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as fun
import torch.utils.data as data


class toolkits(object):
    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray, channel=False):
        assert img1.ndim == 3 and img2.ndim == 3
        img_h, img_w, img_c = img1.shape
        ref = img1.reshape(-1, img_c)
        tar = img2.reshape(-1, img_c)
        msr = np.mean((ref - tar) ** 2, 0)
        if channel is False:
            max2 = np.max(ref) ** 2  # channel-wise ???
        else:
            max2 = np.max(ref, axis=0) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean

    @staticmethod
    def compute_sam(label: np.ndarray, output: np.ndarray):
        h, w, c = label.shape
        x_norm = np.sqrt(np.sum(np.square(label), axis=-1))
        y_norm = np.sqrt(np.sum(np.square(output), axis=-1))
        xy_norm = np.multiply(x_norm, y_norm)
        xy = np.sum(np.multiply(label, output), axis=-1)
        dist = np.mean(np.arccos(np.minimum(np.divide(xy, xy_norm + 1e-8), 1.0 - 1.0e-9)))
        dist = np.multiply(180.0 / np.pi, dist)
        return dist

    @staticmethod
    def check_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def channel_last(input_tensor: np.ndarray, squeeze=True):
        if squeeze is True:
            input_tensor = np.squeeze(input_tensor)
        input_tensor = np.transpose(input_tensor, axes=(1, 2, 0))
        return input_tensor

    @staticmethod
    def channel_first(input_tensor: np.ndarray, expand=True):
        input_tensor = np.transpose(input_tensor, axes=(2, 0, 1))
        if expand is True:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor


class torchkits(object):
    @staticmethod
    def extract_patches(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
        N, C, H, W, h, w = all_patches.shape
        all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)
        all_patches = torch.reshape(all_patches, shape=(N * H * W, C, h, w))
        return all_patches

    @staticmethod
    def torch_norm(input_tensor: torch.Tensor, mode=1):
        if mode == 1:
            loss = torch.sum(torch.abs(input_tensor))
            return loss
        return None

    @staticmethod
    def get_param_num(model):
        num = sum(x.numel() for x in model.parameters())
        print("model has {} parameters in total".format(num))
        return num

    @staticmethod
    def to_numpy(val: torch.Tensor):
        return val.cpu().detach().numpy()


class BlurDown(object):
    def __init__(self, shift_h=0, shift_w=0, stride=0):
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, pad, groups, ratio):
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1)
        if self.stride == 0:
            output_tensor = fun.conv2d(input_tensor, psf, None, (1, 1), (pad, pad), groups=groups)
            output_tensor = output_tensor[:, :, self.shift_h:: ratio, self.shift_h:: ratio]
        else:
            output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio), (pad, pad), groups=groups)
        return output_tensor


class DataInfo(object):
    """
        file structure
        ./data/
        ../data/pavia/
        ../data/moffett/
        ../data/dc/
        .../data/pavia/XXX/
        .../data/pavia/BlindTest/
        .../data/pavia/pavia_data_r?_?_?.mat
        ..../data/pavia/BlindTest/r?_?_?/
        ..../data/pavia/BlindTest/model/
        ..../data/pavia/BlindTest/BR.mat
    """
    def __init__(self, ndata=0, nratio=4, nsnr=0):
        name = self.__class__.__name__
        print('%s is running' % name)
        self.gen_path = 'F:/data_pycode/'  # change
        self.folder_names = ['pavia/', 'ksc/', 'dc/', 'uh/']
        self.data_names = ['pavia_data_r', 'ksc_data_r', 'dc_data_r', 'UH_test_r']
        self.noise = ['_20_30', '_25_35', '_30_40', '_50_60', '']
        self.file_path = self.gen_path + self.folder_names[ndata] + self.data_names[ndata] + str(nratio) + self.noise[
            nsnr] + '.mat'
        mat = sio.loadmat(self.file_path)
        hsi, msi = mat['I_HS'], mat['I_MS']  # h x w x L, H x W x l
        if 'I_REF' in mat.keys():
            ref = mat['I_REF']  # H x W X L
        else:
            ref = np.ones(shape=(msi.shape[0], msi.shape[1], hsi.shape[2]))
        if 'K' in mat.keys():
            psf, srf = mat['K'], mat['R']  # K x K, l X L
        else:
            psf = np.ones(shape=(msi.shape[0] // hsi.shape[0], msi.shape[1] // hsi.shape[1]))
            srf = np.ones(shape=(msi.shape[-1], hsi.shape[-1]))
        self.save_path = self.gen_path + self.folder_names[ndata] + name + '/r' + str(nratio) + self.noise[
            nsnr] + '/'
        hsi = hsi.astype(np.float32)
        msi = msi.astype(np.float32)
        self.ref = ref.astype(np.float32)
        self.psf = psf.astype(np.float32)
        self.srf = srf.astype(np.float32)
        self.model_save_path = self.save_path + 'model/'
        # preprocess
        self.hsi = toolkits.channel_first(hsi)  # 1 x L x h x w
        self.msi = toolkits.channel_first(msi)  # 1 x l x H x W
        self.hs_bands, self.ms_bands = self.hsi.shape[1], self.msi.shape[1]
        self.ratio = int(self.msi.shape[-1] / self.hsi.shape[-1])
        self.height, self.width = self.msi.shape[2], self.msi.shape[3]
        pass


class PatchDataset(data.Dataset):
    # divide the images into several (overlapped) patches
    def __init__(self, hsi: torch.Tensor, msi: torch.Tensor, hsi_up: torch.Tensor, kernel, stride, ratio=1):
        super(PatchDataset, self).__init__()
        self.hsi = torchkits.extract_patches(hsi, kernel // ratio, stride // ratio, pad_num=0 // ratio)
        self.msi = torchkits.extract_patches(msi, kernel, stride, pad_num=0)
        self.hsi_up = torchkits.extract_patches(hsi_up, kernel, stride, pad_num=0)
        self.num = self.msi.shape[0]
        assert self.hsi.shape[0] == self.num

    def __getitem__(self, item):
        hsi = self.hsi[item, :, :, :]
        msi = self.msi[item, :, :, :]
        hsi_up = self.hsi_up[item, :, :, :]
        return hsi, msi, hsi_up, item

    def __len__(self):
        return self.num
