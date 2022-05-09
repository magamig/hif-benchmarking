import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)


        self.GT = dataset.get("GT")
        # self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")




    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.GT[index, :, :, :]).float(), \
               torch.from_numpy(self.LRHSI[index, :, :, :]).float(), \
               torch.from_numpy(self.RGB[index, :, :, :]).float()


                   #####必要函数
    def __len__(self):
        return self.GT.shape[0]
