import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image

DATASET = 'EHU'
SCALINGS = ([int(sys.argv[1])] if len(sys.argv) >= 2 else [4,8,16])
GT_PATH = f'data/GT/{DATASET}'
MS_PATH = f'data/MS/{DATASET}'
HS_PATH = f'data/HS/{DATASET}'
os.makedirs(MS_PATH, exist_ok = True)
for sf in SCALINGS:
    os.makedirs(f'{HS_PATH}/{sf}', exist_ok = True)

os.makedirs(GT_PATH, exist_ok = True)
os.makedirs(MS_PATH, exist_ok = True)
os.makedirs(HS_PATH, exist_ok = True)
if not os.path.exists(f"{GT_PATH}/Indian_pines.mat"):
    os.system(f"wget http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat -P {GT_PATH}")
if not os.path.exists(f"{GT_PATH}/KSC.mat"):
    os.system(f"wget http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat -P {GT_PATH}")
if not os.path.exists(f"{GT_PATH}/Salinas.mat"):
    os.system(f"wget http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat -P {GT_PATH}")
if not os.path.exists(f"{GT_PATH}/SalinasA.mat"):
    os.system(f"wget http://www.ehu.eus/ccwintco/uploads/d/df/SalinasA.mat -P {GT_PATH}")
if not os.path.exists(f"{GT_PATH}/Cuprite.mat"):
    os.system(f"wget http://www.ehu.eus/ccwintco/uploads/7/7d/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat -O {GT_PATH}/Cuprite.mat")
if not os.path.exists(f"{GT_PATH}/Botswana.mat"):
    os.system(f"wget http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat -P {GT_PATH}")
if not os.path.exists(f"{GT_PATH}/Pavia.mat"):
    os.system(f"wget http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat -P {GT_PATH}")
if not os.path.exists(f"{GT_PATH}/PaviaU.mat"):
    os.system(f"wget http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat -P {GT_PATH}")

 # downsampling HS image
for mat_path in glob.iglob(f'{GT_PATH}/*.mat'):
    name = Path(mat_path).stem
    print(name)
    mat = scipy.io.loadmat(mat_path)
    hsi = mat[list(mat.keys())[-1]]
    hsi = hsi.astype("float32") / hsi.max()
    for sf in SCALINGS:
        hsi_downsampled = None
        for i in range(hsi.shape[2]):
            # from np to Image
            img = Image.fromarray(hsi[:,:,i])
            img = img.resize((hsi.shape[1]//sf, hsi.shape[0]//sf), Image.LANCZOS)
            # from Image to np
            img = np.expand_dims(np.asarray(img), axis=2)
            hsi_downsampled = img if  hsi_downsampled is None else np.concatenate((hsi_downsampled, img), axis=2)
        scipy.io.savemat(f'{HS_PATH}/{sf}/{name}.mat', {"hsi": hsi_downsampled})

os.system(f'''/Applications/MATLAB_R20*.app/bin/matlab -nojvm -nodesktop -nodisplay -nosplash -batch "warning('off'); addpath(genpath('main')); run('aux_EHU'); exit;"''')
