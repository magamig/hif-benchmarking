import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image

DATASET = 'Harvard'
DOWNSAMPLE = int(sys.argv[1] if len(sys.argv) >= 2 else 4)
GT_PATH = f'data/GT/{DATASET}'
MS_PATH = f'data/MS/{DATASET}'
HS_PATH = f'data/HS/{DATASET}/{DOWNSAMPLE}'
T = np.array([
    [0.005,0.007,0.012,0.015,0.023,0.025,0.030,0.026,0.024,0.019,0.010,0.004,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.000,0.000,0.000,0.000,0.000,0.001,0.002,0.003,0.005,0.007,0.012,0.013,0.015,0.016,0.017,0.02,0.013,0.011,0.009,0.005,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.002,0.002,0.003],
    [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.001,0.003,0.010,0.012,0.013,0.022,0.020,0.020,0.018,0.017,0.016,0.016,0.014,0.014,0.013]
]).T

os.system("wget http://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdbi.tgz")
os.system("wget http://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdb.tgz")
os.system("tar -xvzf CZ_hsdbi.tgz -C data/GT/")
os.system("tar -xvzf CZ_hsdb.tgz -C data/GT/")
os.makedirs(GT_PATH, exist_ok = True)
os.system(f"mv data/GT/CZ_hsdbi/* {GT_PATH}")
os.system(f"mv data/GT/CZ_hsdb/* {GT_PATH}")
os.system("rm -r data/GT/CZ_hsdbi")
os.system("rm -r data/GT/CZ_hsdb")
os.system("rm CZ_hsdbi.tgz")
os.system("rm CZ_hsdb.tgz")
os.makedirs(MS_PATH, exist_ok = True)
os.makedirs(HS_PATH, exist_ok = True)

for mat_path in glob.iglob(f'{GT_PATH}/*.mat'):
    print(name)
    name = Path(mat_path).stem
    mat = scipy.io.loadmat(mat_path)
    hsi = mat['ref']
    # downsampling HS image
    hsi_downsampled = None
    for i in range(hsi.shape[2]):
        # from np to Image
        img = Image.fromarray(hsi[:,:,i])
        img = img.resize((hsi.shape[0]//DOWNSAMPLE, hsi.shape[1]//DOWNSAMPLE), Image.LANCZOS)
        # from Image to np
        img = np.expand_dims(np.asarray(img), axis=2)
        hsi_downsampled = img if  hsi_downsampled is None else np.concatenate((hsi_downsampled, img), axis=2)
    scipy.io.savemat(f'{HS_PATH}/{name}.mat', {"hsi": hsi_downsampled})
    # simulate RGB photo with Nikon D700 camera
    msi = np.dot(hsi,T)
    scipy.io.savemat(f'{MS_PATH}/{name}.mat', {"msi": msi})
