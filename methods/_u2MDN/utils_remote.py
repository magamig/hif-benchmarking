import numpy as np
import scipy.io as sio
import collections
import scipy.misc


# Written by Ying Qu, <yqu3@vols.utk.edu>
# This is a demo code for 'Unsupervised and Unregistered Hyperspectral Image Super-Resolution with Mutual Dirichlet-Net'. 
# The code is for research purposes only. All rights reserved.


def loadData(mname):
    return sio.loadmat(mname)

def readData(filename,num=10):
    # load hsi, msi and ref from .mat files,all the images should be normolized between [0,1] first.
    input = loadData(filename)
    data = collections.namedtuple('data', ['HSI', 'MSI', 'REF', 'srf', 'scaling','ratio'])
    data.scaling = input['scaling'].astype(np.float32)
    data.hyperLR = np.array(input['HSI']).astype(np.float32)/data.scaling
    data.multiHR = np.array(input['MSI']).astype(np.float32)/data.scaling
    data.hyperHR = np.array(input['REF']).astype(np.float32)/data.scaling
    data.srf = np.array(input['srf']).astype(np.float32).T
    data.dimLR = data.hyperLR.shape
    data.dimHR = data.multiHR.shape
    data.num = num
    data.srfactor = np.divide(data.dimHR[0],data.dimLR[0])

    data.hsi_org_rd = data.hyperLR
    data.dimLR_hsi_lr = data.hsi_org_rd.shape
    data.colhsi_lr = np.reshape(data.hsi_org_rd, [data.dimLR_hsi_lr[0] * data.dimLR_hsi_lr[1], data.dimLR_hsi_lr[2]])
    data.meanhsi_lr  = np.mean(data.colhsi_lr, axis=0, keepdims=True)
    data.patch_hsi_lr = np.subtract(data.hsi_org_rd, data.meanhsi_lr)

    data.hsi_org = data.hyperLR
    data.dimLR_hsi = data.hyperLR.shape
    data.colhsi = np.reshape(data.hsi_org, [data.dimLR_hsi[0] * data.dimLR_hsi[1], data.dimLR_hsi[2]])
    data.meanhsi  = np.mean(data.colhsi, axis=0, keepdims=True)
    data.patch_hsi = np.subtract(data.hsi_org, data.meanhsi)

    data.msi_org = data.multiHR
    data.dimHR_msi = data.msi_org.shape
    data.colmsi = np.reshape(data.msi_org, [data.dimHR_msi[0] * data.dimHR_msi[1], data.dimHR_msi[2]])
    data.meanmsi = np.mean(data.colmsi, axis=0, keepdims=True)
    data.patch_msi = np.subtract(data.msi_org, data.meanmsi)
    data.patch_hr = data.hyperHR

    return data
