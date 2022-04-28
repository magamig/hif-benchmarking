import glob
import os
import warnings
from pathlib import Path

import numpy as np
import scipy.io
from scipy.ndimage import uniform_filter
from sewar.full_ref import (msssim, psnr, psnrb, rmse, rmse_sw, sam, scc, ssim,
                            uqi, vifp)
from sewar.utils import _initial_check

warnings.filterwarnings('ignore')

GT_PATH = f'data/GT'
SR_PATH = f'data/SR'


def ergas(GT,P,r=4,ws=8):
    ''' sewar.full_ref.ergas but survives NaNs '''
    GT,P = _initial_check(GT,P)

    rmse_map = None
    nb = 1

    _,rmse_map = rmse_sw(GT,P,ws)

    means_map = uniform_filter(GT,ws)/ws**2

    # Avoid division by zero
    idx = means_map == 0
    means_map[idx] = 1
    rmse_map[idx] = 0

    ergasroot = np.sqrt(np.sum(((rmse_map**2)/(means_map**2)),axis=2)/nb)
    ergas_map = 100*r*ergasroot

    s = int(np.round(ws/2))
    return np.nanmean(ergas_map[s:-s,s:-s])


def rase(GT,P,ws=8):
    ''' sewar.full_ref.rase but survives NaNs '''

    GT,P = _initial_check(GT,P)

    _,rmse_map = rmse_sw(GT,P,ws)

    GT_means = uniform_filter(GT, ws)/ws**2

    N = GT.shape[2]
    M = np.sum(GT_means,axis=2)/N
    rase_map = (100./M) * np.sqrt( np.sum(rmse_map**2,axis=2) / N )

    s = int(np.round(ws/2))
    return np.nanmean(rase_map[s:-s,s:-s])


def main():
    method_paths = glob.glob(f"{SR_PATH}/*")
    for cm, method_path in enumerate(method_paths, start=1):
        method = Path(method_path).stem
        
        dataset_paths = glob.glob(f"{method_path}/*")
        for cd, dataset_path in enumerate(dataset_paths, start=1):
            dataset = Path(dataset_path).stem
            
            scaling_paths = glob.glob(f"{dataset_path}/*")
            for cs, scaling_path in enumerate(scaling_paths, start=1):
                csv_path = f"{scaling_path}/metrics.csv"
                scaling = Path(scaling_path).stem

                if not os.path.exists(csv_path):
                    csv = "name,rmse,psnr,psnrb,ssim,msssim,uqi,ergas,scc,rase,sam,vifp\n"
                    mat_paths = glob.glob(f"{scaling_path}/*.mat")
                    for ci, mat_path in enumerate(mat_paths, start=1):
                        name = Path(mat_path).stem
                        print(f"method: {method} ({cm}/{len(method_paths)}), dataset: {dataset} ({cd}/{len(dataset_paths)}), scale: {scaling} ({cs}/{len(scaling_paths)}), img: {name}({ci}/{len(mat_paths)})")
                        
                        sri = scipy.io.loadmat(mat_path)["sri"]
                        if  np.issubdtype(sri.dtype, np.integer):
                            sri = np.float64(sri) / np.iinfo(sri.dtype).max
                        hsi = scipy.io.loadmat(f"{GT_PATH}/{dataset}/{name}.mat")["hsi"]
                        if  np.issubdtype(hsi.dtype, np.integer):
                            hsi = np.float64(hsi) / np.iinfo(hsi.dtype).max
                        
                        metrics = [
                            rmse(hsi, sri) * 100,
                            psnr(hsi, sri, MAX=1.0),
                            psnrb(hsi, sri),
                            ssim(hsi, sri, MAX=1.0)[0],
                            float(msssim(hsi, sri, MAX=1.0)),
                            uqi(hsi, sri),
                            ergas(hsi, sri),
                            scc(hsi, sri),
                            rase(hsi, sri),
                            sam(hsi, sri),
                            vifp(hsi, sri),
                        ]
                        csv += f"{name},{','.join(str(e) for e in metrics)}\n"
                        print(csv)
                    f = open(csv_path, "w+")
                    f.write(csv)
                    f.close()

        
if __name__ == "__main__":
    main()
