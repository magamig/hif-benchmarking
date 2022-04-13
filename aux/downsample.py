import glob
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

DATASET = sys.argv[1]
DOWNSAMPLE = int(sys.argv[2] or 4)
GT_DIR = f'data/GT/{DATASET}'
RGB_DIR = f'data/RGB/{DATASET}'
HSI_DIR = f'data/HSI/{DATASET}/{DOWNSAMPLE}'

os.makedirs(RGB_DIR, exist_ok = True)
os.makedirs(HSI_DIR, exist_ok = True)

for mat_path in glob.iglob(f'{GT_DIR}/*.mat'):
    hscube_name = Path(mat_path).stem
    mat = scipy.io.loadmat(mat_path)
    rgb = mat['rgb']
    hsi = mat['hsi']
    # saving RGB
    scipy.io.savemat(f'{RGB_DIR}/{hscube_name}.mat', {"rgb": rgb})
    # downsampling HS image
    hscube = None
    for i in range(hsi.shape[2]):
        # from np to Image
        img = Image.fromarray(hsi[:,:,i])
        img = img.resize((hsi.shape[0]//DOWNSAMPLE, hsi.shape[1]//DOWNSAMPLE), Image.LANCZOS)
        # from Image to np
        img = np.expand_dims(np.asarray(img), axis=2)
        hscube = img if  hscube is None else np.concatenate((hscube, img), axis=2)
    scipy.io.savemat(f'{HSI_DIR}/{hscube_name}.mat', {"hsi": hscube})
