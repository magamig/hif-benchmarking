import argparse
import glob
import os
import sys
import time
from pathlib import Path

DATASETS = [
    "ResolutionChart",
    "CAVE",
    "EHU",
    "Harvard",
]
ALL_METHODS = [
    "CNMF",
    "FUSE",
    "SFIM",
    "GSA",
    "GLP",
    "GSOMP",
    "NSSR",
    "SupResPALM",
    "CNNFUS",
    "HySure", # slow
    "MAPSMM", # slow
    "LTTR", # slow
    "LTMR", # slow
    # "CSTF", # unstable - needs fixes
    # "BayesianSparse", # very slow
]
# problems in papers methods:
# - custom downsampling parameters according to method;
# - use of the kernel blurring data as input (used to generate HS from GT)
# - the use of fake matrix T as input (used to generate MS from GT)
# - different downsampling techniques used (to generate HS from GT)
# - some use noise some don't
# - use different datasets
# - existing comparison review only uses remote sensing datasets and are old 
# - no usage of denoising of GT to increase SNRs of reference images [Naoto Yokoya review]
# - error in corregistration not taken into account
METHODS_PAPER_DIFF = [
    # "CNMF", # OK
    "FUSE", # uses matrix T and kernel info as input
    # "SFIM", # OK (via hypershaperning)
    # "GSA", # OK
    # "GLP", # OK (via hypershaperning)
    "GSOMP", # uses matrix T and downsample using matlab function "downsample"
    "NSSR", # uses matrix T and kernel info as input with custom parameters accordingly (uniform & gaussian kernels)
    "SupResPALM", # uses matrix T
    "CNNFUS", # uses matrix T and kernel info as input (noise from demo not used)
    # "HySure", # OK but removes noisy bands beforehand
    # "MAPSMM", # OK
    "LTTR", # uses matrix T & removes noisy bands beforehand
    "LTMR", # uses matrix T & removes noisy bands beforehand
]
SCALES = [4]


def matlabcmd(cmd, path="."):
    if sys.platform == "darwin":
        os.system(f'''/Applications/MATLAB_R20*.app/bin/matlab -nojvm -nodesktop -nodisplay -nosplash -batch "warning('off'); addpath(genpath('aux')); addpath(genpath('{path}')); {cmd}; exit;"''')
    elif sys.platform == "win32":
        os.system(f'''matlab.exe -nojvm -nodesktop -nosplash -batch "warning('off'); addpath(genpath('aux')); addpath(genpath('{path}')); {cmd}; exit;"''')
    elif sys.platform.startswith("linux"):  # could be "linux", "linux2", "linux3", ...
        raise Exception("Script for linux not implemented yet.")

def get_paths(dataset, method, scale, img, run_as_papers=False):
    hsi_path = f"data/HS/{dataset}/{scale}/{img}.mat"
    msi_path = f"data/MS/{dataset}/{img}.mat"
    gti_path = f"data/GT/{dataset}/{img}.mat"
    sr_path = f"data/{'paper_' if run_as_papers else ''}SR/{method}/{dataset}/{scale}"
    os.makedirs(sr_path, exist_ok = True)
    sri_path = f"{sr_path}/{img}.mat"
    return hsi_path, msi_path, sri_path, gti_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", help="run methods as in the papers", action='store_true')
    args = parser.parse_args()
    run_as_papers = args.paper
    methods = METHODS_PAPER_DIFF if run_as_papers else ALL_METHODS
    
    for cd, dataset in enumerate(DATASETS, start=1):
        img_paths = glob.glob(f'data/GT/{dataset}/*.mat')
        for cm, method in enumerate(methods, start=1):
            for cs, scale in enumerate(SCALES, start=1):
                for ci, img_path in enumerate(img_paths, start=1):
                    img = Path(img_path).stem
                    print(f"dataset: {dataset} ({cd}/{len(DATASETS)}), method: {method} ({cm}/{len(methods)}), scale: {scale} ({cs}/{len(SCALES)}), img: {img}({ci}/{len(img_paths)})")
                    hsi_path, msi_path, sri_path, gti_path = get_paths(dataset, method, scale, img, run_as_papers)
                    if not os.path.exists(sri_path):
                        tic = time.time()
                        if not run_as_papers:
                            cmd = f'''hsi_path='{hsi_path}';msi_path='{msi_path}';sri_path='{sri_path}';{method}_run'''
                        else:
                            cmd = f'''gti_path='{gti_path}';sri_path='{sri_path}';scale={scale};{method}_paper_run'''
                        matlabcmd(cmd, f"methods/{method}")
                        toc = time.time()
                        f = open(f'{os.path.dirname(sri_path)}/times.csv', "a+")
                        f.write(f'{img},{toc-tic}\n')
                        f.close()
                        


if __name__ == "__main__":
    main()
