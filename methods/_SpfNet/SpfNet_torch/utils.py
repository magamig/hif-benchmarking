# Author: JianJun Liu
# Date: 2021/12/27
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
import torch
import torch.nn.functional as fun
import torch.utils.data as data


def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


class toolkits:
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
    def psnr_fun(ref: np.ndarray, tar: np.ndarray, max_value=None):
        assert ref.ndim == 4 and tar.ndim == 4
        b, c, h, w = ref.shape
        ref = ref.reshape(b, c, h * w)
        tar = tar.reshape(b, c, h * w)
        msr = np.mean((ref - tar) ** 2, 2)
        max2 = np.max(ref, axis=2) ** 2
        if max_value is not None:
            max2 = max_value
        psnrall = 10 * np.log10(max2 / msr)
        return np.mean(psnrall)

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


class torchkits:
    @staticmethod
    def sparse_to_torch(input_tensor: sp.coo_matrix):
        values = input_tensor.data
        indices = np.vstack((input_tensor.row, input_tensor.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = input_tensor.shape
        input_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return input_tensor

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
    def extract_patches_v1(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        N, C, H, W = input_tensor.shape
        unfold = torch.nn.Unfold(kernel_size=(kernel, kernel), stride=stride)
        all_patches = unfold(input_tensor)
        _, _, L = all_patches.shape
        all_patches = torch.reshape(all_patches, shape=(N, C, kernel, kernel, L))
        all_patches = all_patches.permute(0, 4, 1, 2, 3)
        all_patches = torch.reshape(all_patches, shape=(N * L, C, kernel, kernel))
        return all_patches

    @staticmethod
    def extract_patches_ex(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
        # N, C, H, W, h, w = all_patches.shape
        all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)  # shape=(N, H, W, C, h, w)
        return all_patches

    @staticmethod
    def aggregate_patches(input_tensor: torch.Tensor, height, width, kernel, stride, pad_num=0, patch=1):
        N, C, h, w = input_tensor.shape
        dH = height + 2 * pad_num - (height + 2 * pad_num - kernel) // stride * stride - kernel
        dW = width + 2 * pad_num - (width + 2 * pad_num - kernel) // stride * stride - kernel
        height, width = height - dH, width - dW
        input_tensor = input_tensor.reshape(patch, N // patch, C, h, w)
        output_tensor = input_tensor.permute(0, 2, 3, 4, 1)
        output_tensor = torch.reshape(output_tensor, shape=(patch, C * h * w, N // patch))
        num = torch.ones_like(output_tensor)
        fold = torch.nn.Fold(output_size=(height + 2 * pad_num, width + 2 * pad_num),
                             kernel_size=(kernel, kernel),
                             stride=stride)
        output_tensor = fold(output_tensor)
        num = fold(num)
        output_tensor = output_tensor[:, :, pad_num: height + pad_num, pad_num: width + pad_num]
        num = num[:, :, pad_num: height + pad_num, pad_num: width + pad_num]
        output_tensor = output_tensor / num
        return output_tensor, dH, dW

    @staticmethod
    def torch_norm(input_tensor: torch.Tensor, mode=1):
        if mode == 1:
            loss = torch.sum(torch.abs(input_tensor))
            return loss
        return None

    @staticmethod
    def torch_sam(label: torch.Tensor, output: torch.Tensor):
        x_norm = torch.sqrt(torch.sum(torch.square(label), dim=-1))
        y_norm = torch.sqrt(torch.sum(torch.square(output), dim=-1))
        xy_norm = torch.multiply(x_norm, y_norm)
        xy = torch.sum(torch.multiply(label, output), dim=-1)
        dist = torch.divide(xy, xy_norm + torch.tensor(1e-8))
        dist = torch.mean(torch.arccos(torch.minimum(dist, torch.tensor(1.0 - 1e-9))))
        dist = torch.multiply(torch.tensor(180.0 / np.pi), dist)
        return dist

    @staticmethod
    def sparsity_l1_div_l2(x: torch.Tensor):
        N, C, H, W = x.shape  # perform on mode-C
        l1norm = torch.sum(torch.abs(x), dim=1)
        l2norm = torch.sqrt(torch.sum(torch.square(x), dim=1))
        sparsity = torch.sum(l1norm / l2norm)
        return sparsity

    @staticmethod
    def joint_sparsity(x: torch.Tensor):
        N, H, W = x.shape  # perform on mode H, W
        l2norm = torch.sqrt(torch.sum(torch.square(x), dim=2))
        l21norm = torch.sum(l2norm, dim=1)
        fnorm = torch.sqrt(torch.sum(torch.square(x), dim=(1, 2))) + 1e-9
        return torch.sum(l21norm / fnorm)

    @staticmethod
    def sp_joint_l1_div_l2(img: torch.Tensor, jdx: torch.Tensor):
        _, C, W, H = img.shape
        output = torch.squeeze(img)
        output = torch.reshape(output, shape=(C, W * H))
        output = torch.transpose(output, 0, 1)
        output = torch.square(output)
        output = torch.matmul(jdx, output)
        l1norm = torch.sum(torch.sqrt(output), dim=1)
        l2norm = torch.sum(output, dim=1)
        output = torch.sum(l1norm / l2norm)
        return output

    @staticmethod
    def sp_joint_l21(img: torch.Tensor, jdx: torch.Tensor):
        _, C, W, H = img.shape
        output = torch.squeeze(img)
        output = torch.reshape(output, shape=(C, W * H))
        output = torch.transpose(output, 0, 1)
        output = torch.square(output)
        output = torch.matmul(jdx, output)
        output = torch.sqrt(output)
        output = torch.sum(output)
        return output

    @staticmethod
    def superpixel_mean(img: torch.Tensor, jdx: torch.Tensor, jdx_n: torch.Tensor):
        _, C, W, H = img.shape
        output_tensor = torch.squeeze(img)
        output_tensor = torch.reshape(output_tensor, shape=(C, W * H))
        output_tensor = torch.transpose(output_tensor, 0, 1)
        output_tensor = torch.matmul(jdx_n, output_tensor)
        output_tensor = torch.matmul(jdx, output_tensor)
        output_tensor = torch.transpose(output_tensor, 0, 1)
        output_tensor = torch.reshape(output_tensor, shape=(1, C, W, H))
        return output_tensor

    @staticmethod
    def get_param_num(model):
        num = sum(x.numel() for x in model.parameters())
        print("model has {} parameters in total".format(num))
        return num

    @staticmethod
    def to_numpy(val: torch.Tensor):
        return val.cpu().detach().numpy()


class ttorch:
    @staticmethod
    def kr(a: torch.Tensor, b: torch.Tensor):
        """
        Khatri–Rao Product
        """
        I, R = a.shape
        J, R = b.shape
        c = torch.einsum('ir,jr->ijr', a, b)
        c = torch.reshape(c, shape=(I * J, R))
        return c

    @staticmethod
    def cp(a, b, c):
        I, R = a.shape
        J, R = b.shape
        K, R = c.shape
        ab = ttorch.kr(a, b)
        abc = torch.matmul(ab, c.t())
        abc = torch.reshape(abc, shape=(I, J, K))
        return abc

    @staticmethod
    def cp_(a, b, c):
        """
        pytorch einsum is not good!
        """
        I, R = a.shape
        J, R = b.shape
        K, R = c.shape
        abc = torch.einsum('ir,jr,kr->ijk', a, b, c)
        return abc


class BlurDown:
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


class DataInfo:
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


class PatchMemDataset(data.Dataset):
    # less memory version
    def __init__(self, hsi: torch.Tensor, msi: torch.Tensor, hsi_up: torch.Tensor, kernel, stride, ratio=1):
        super(PatchMemDataset, self).__init__()
        self.hsi = torch.squeeze(hsi, dim=0)
        self.msi = torch.squeeze(msi, dim=0)
        self.hsi_up = torch.squeeze(hsi_up, dim=0)
        self.kernel = kernel
        self.stride = stride
        self.ratio = ratio
        _, self.height, self.width = self.msi.shape
        self.h_num = (self.height - self.kernel) // self.stride + 1
        self.w_num = (self.width - self.kernel) // self.stride + 1
        self.num = self.h_num * self.w_num

    def __getitem__(self, item):
        pos_h = item // self.w_num
        pos_w = item - pos_h * self.w_num
        W = pos_w * self.stride
        H = pos_h * self.stride
        w = W // self.ratio
        h = H // self.ratio
        hsi = self.hsi[:, h: h + self.kernel // self.ratio, w: w + self.kernel // self.ratio]
        msi = self.msi[:, H: H + self.kernel, W: W + self.kernel]
        hsi_up = self.hsi_up[:, H: H + self.kernel, W: W + self.kernel]
        return hsi, msi, hsi_up, item

    def __len__(self):
        return self.num


class WinDataset(data.Dataset):
    # divide the images into patches so that one can extract local window for each pixel
    def __init__(self, hsi: torch.Tensor, msi: torch.Tensor, hsi_up: torch.Tensor, kernel, stride, ratio, win=5):
        super(WinDataset, self).__init__()
        self.hsi = torch.squeeze(hsi, dim=0)  # C, h, w
        self.msi = torch.squeeze(msi, dim=0)  # c, H, W
        self.kernel = kernel
        self.stride = stride
        self.ratio = ratio
        _, self.height, self.width = self.msi.shape
        self.h_num = (self.height - self.kernel) // self.stride + 1
        self.w_num = (self.width - self.kernel) // self.stride + 1
        self.num = self.h_num * self.w_num
        self.win = win
        self.win_pad = win // 2
        fun_pad = torch.nn.ReflectionPad2d(self.win_pad)
        self.msi_pad = fun_pad(msi)
        self.hsi_up_pad = fun_pad(hsi_up)
        self.msi_pad = torch.squeeze(self.msi_pad, dim=0)  # c, H+win, W+win
        self.hsi_up_pad = torch.squeeze(self.hsi_up_pad, dim=0)   # C, H+win, W+win

    def __getitem__(self, item):
        pos_h = item // self.w_num
        pos_w = item - pos_h * self.w_num
        W = pos_w * self.stride
        H = pos_h * self.stride
        w = W // self.ratio
        h = H // self.ratio
        hsi = self.hsi[:, h: h + self.kernel // self.ratio, w: w + self.kernel // self.ratio]
        msi = self.msi[:, H: H + self.kernel, W: W + self.kernel]
        msi_pad = self.msi_pad[:, H: H + self.kernel + 2 * self.win_pad, W: W + self.kernel + 2 * self.win_pad]
        hsi_up_pad = self.hsi_up_pad[:, H: H + self.kernel + 2 * self.win_pad, W: W + self.kernel + 2 * self.win_pad]
        return hsi, msi, msi_pad, hsi_up_pad, item

    def __len__(self):
        return self.num

