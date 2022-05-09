import numpy as np
import scipy.io as sio
import collections
import scipy.misc


def loadData(mname):
    return sio.loadmat(mname)

def readData(filename,num=10):
    input = loadData(filename)
    data = collections.namedtuple('data', ['hyperLR', 'multiHR', 'hyperHR',
                                           'dimLR', 'dimHR', 'srf','srfactor',
                                           'colLR','meanLR', 'reducedLR',
                                           'sphere','num'],
                                   verbose=False)

    data.hyperLR = np.array(input['hyperLR']).astype(np.float32)
    data.multiHR = np.array(input['multiHR']).astype(np.float32)
    data.hyperHR = np.array(input['hyperHR']).astype(np.float32)
    data.hyperLRI = np.array(input['hyperLRI']).astype(np.float32)
    data.dimLR = data.hyperLR.shape
    data.dimHR = data.multiHR.shape
    data.num = num

    # 3*31
    srf = [[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
            0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
            0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
           [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
            0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]]

    srf = np.array(srf).astype(np.float32)
    data.srf = srf.astype(np.float32)
    data.sr_factor = np.divide(data.dimHR[0],data.dimLR[0]).astype(np.int)

    data.col_lr_hsi = np.reshape(data.hyperLR,[data.dimLR[0]*data.dimLR[1],data.dimLR[2]])
    data.mean_lr_hsi = np.mean(data.col_lr_hsi,axis=0,keepdims=True)
    data.rcol_lr_hsi = np.subtract(data.col_lr_hsi,data.mean_lr_hsi)
    data.img_lr_hsi = np.reshape(data.rcol_lr_hsi,[data.dimLR[0],data.dimLR[1],data.dimLR[2]])

    data.col_hr_msi = np.reshape(data.multiHR, [data.dimHR[0] * data.dimHR[1], data.dimHR[2]])
    data.mean_hr_msi = np.mean(data.col_hr_msi, axis=0, keepdims=True)
    data.rcol_hr_msi = np.subtract(data.col_hr_msi,data.mean_hr_msi)
    data.img_hr_msi = np.reshape(data.rcol_hr_msi, [data.dimHR[0],data.dimHR[1],data.dimHR[2]])

    data.multiLR = scipy.ndimage.zoom(data.multiHR, zoom=[1.0 / data.sr_factor, 1.0 / data.sr_factor, 1],order=0)
    data.col_lr_msi = np.reshape(data.multiLR, [data.dimLR[0] * data.dimLR[1], data.dimHR[2]])
    data.mean_lr_msi = np.mean(data.col_lr_msi, axis=0, keepdims=True)
    data.rcol_lr_msi = np.subtract(data.col_lr_msi,data.mean_lr_msi)


    data.col_hr_hsi = np.reshape(data.hyperHR,[data.dimHR[0]*data.dimHR[1],data.dimLR[2]])

    return data