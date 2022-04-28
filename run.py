import argparse
import glob
import os
from pathlib import Path

DATASETS = ["CAVE"]
ALL_METHODS = ["LTMR","CNMF","FUSE","SFIM","GSA","GLP","GSOMP","NSSR","SupResPALM","LTTR","HySure","MAPSMM","CSTF","BayesianSparse"]
METHODS_PAPER_DIFF = [ # custom downsampling methods according to method
    # "LTMR", # OK 
    # "CNMF", # OK 
    "FUSE", # uses matrix T and kernel info as input
    # "SFIM", # OKOK
    # "GSA", # OKOK
    # "GLP", # OK
    "GSOMP", # uses matrix T and downsample using matlab function "downsample"
    "NSSR", # uses matrix T and kernel info as input with custom parameters accordingly (uniform & gaussian kernels)
    "SupResPALM", # uses matrix T
    "LTTR", # removes noisy bands beforehand
    # "HySure", # removes noisy bands beforehand, uses matrix T and kernel info as input 
    # "MAPSMM", # OKOK
    # "BayesianSparse"
]
SCALES = [4]


def matlabcmd(cmd, path="."):
    os.system(f'''/Applications/MATLAB_R20*.app/bin/matlab -nojvm -nodesktop -nodisplay -nosplash -batch "addpath(genpath('aux')); addpath(genpath('{path}')); {cmd}; exit;"''')


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
                        if not run_as_papers:
                            cmd = f'''hsi_path='{hsi_path}';msi_path='{msi_path}';sri_path='{sri_path}';{method}_run'''
                        else:
                            cmd = f'''gti_path='{gti_path}';sri_path='{sri_path}';scale={scale};{method}_paper_run'''
                        matlabcmd(cmd, f"methods/{method}")


if __name__ == "__main__":
    main()
