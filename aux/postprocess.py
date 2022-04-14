import glob
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

DATASET = sys.argv[1]
DOWNSAMPLE = int(sys.argv[2] or 4)
GT_PATH = f'data/GT/{DATASET}'
MS_PATH = f'data/MS/{DATASET}'
HS_PATH = f'data/HS/{DATASET}/{DOWNSAMPLE}'

os.makedirs(MS_PATH, exist_ok = True)
os.makedirs(HS_PATH, exist_ok = True)

for mat_path in glob.iglob(f'{GT_PATH}/*.mat'):
    name = Path(mat_path).stem
    mat = scipy.io.loadmat(mat_path)
    msi = mat['msi']
    hsi = mat['hsi']
    # saving RGB
    scipy.io.savemat(f'{MS_PATH}/{name}.mat', {"msi": msi})
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
