import glob
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

DATASET_DIR = 'data/GT/CAVE'

for hscube_dir in glob.iglob(f'{DATASET_DIR}/*'):
    hscube_name = Path(hscube_dir).stem
    hscube = None
    # read all PNGs correspnding to different spectra to form the HS cube
    for img_path in glob.iglob(f'{hscube_dir}/*.png'):
        img = np.asarray(Image.open(img_path).convert('L'))
        img = np.expand_dims(img, axis=2) 
        hscube = img if  hscube is None else np.concatenate((hscube, img), axis=2)
    # read BMP with the RGB image
    rgb_path = glob.glob(f'{hscube_dir}/*.bmp')[0]
    rgb = np.asarray(Image.open(rgb_path))
    # save both together as MAT file
    scipy.io.savemat(f'{DATASET_DIR}/{hscube_name}.mat', {"hsi": hscube, "rgb": rgb})
