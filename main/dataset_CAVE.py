import glob
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import scipy.io
from PIL import Image

DATASET = 'CAVE'
DOWNSAMPLE = int(sys.argv[1] if len(sys.argv) >= 2 else 4)
GT_PATH = f'data/GT/{DATASET}'
MS_PATH = f'data/MS/{DATASET}'
HS_PATH = f'data/HS/{DATASET}/{DOWNSAMPLE}'

if not os.path.exists("complete_ms_data.zip"):
    os.system("wget https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip")
os.system("unzip complete_ms_data -d data/GT/aux/")
os.makedirs(GT_PATH, exist_ok = True)
os.system(f"cp -r data/GT/aux/*/* {GT_PATH}/")
os.system("rm -r data/GT/aux/")
#os.system("rm complete_ms_data.zip")

for hs_path in glob.iglob(f"{GT_PATH}/*/"):
    name = Path(hs_path).stem
    hsi = None
    # read all PNGs correspnding to different spectra to form the HS cube
    for img_path in glob.iglob(f'{hs_path}/*.png'):
        img = np.asarray(cv.imread(img_path, cv.IMREAD_GRAYSCALE))
        img = np.expand_dims(img, axis=2) 
        hsi = img if  hsi is None else np.concatenate((hsi, img), axis=2)
    # read BMP with the RGB image
    msi_path = glob.glob(f'{hs_path}/*.bmp')[0]
    msi = np.asarray(cv.cvtColor(cv.imread(msi_path), cv.COLOR_BGR2RGB))
    # save both together as MAT file
    scipy.io.savemat(f'{GT_PATH}/{name}.mat', {"hsi": hsi, "msi": msi})

os.system(f"rm -r {GT_PATH}/*/")
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
        img = img.resize((hsi.shape[0]//DOWNSAMPLE, hsi.shape[1]//DOWNSAMPLE), Image.Resampling.LANCZOS)
        # from Image to np
        img = np.expand_dims(np.asarray(img), axis=2)
        hsi_downsampled = img if  hsi_downsampled is None else np.concatenate((hsi_downsampled, img), axis=2)
    scipy.io.savemat(f'{HS_PATH}/{name}.mat', {"hsi": hsi_downsampled})
