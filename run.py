import glob
import os
from pathlib import Path

DATASETS = ["CAVE"]
METHODS = ["CNMF","FUSE","SFIM","GSA","GLP","SupResPALM","HySure","MAPSMM"]
SCALES = [4]

def matlabrun(cmd):
    os.system(f'''/Applications/MATLAB_R20*.app/bin/matlab -nojvm -nodesktop -nodisplay -nosplash -batch "run('{cmd}'), exit;";''')

def matlabcmd(cmd):
    os.system(f'''/Applications/MATLAB_R20*.app/bin/matlab -nojvm -nodesktop -nodisplay -nosplash -batch "addpath(genpath('methods')); {cmd}; exit;"''')

def get_paths(dataset, method, scale, img):
    hsi_path = f"data/HS/{dataset}/{scale}/{img}.mat"
    msi_path = f"data/MS/{dataset}/{img}.mat"
    sr_path = f"data/SR/{method}/{dataset}/{scale}"
    os.makedirs(sr_path, exist_ok = True)
    sri_path = f"{sr_path}/{img}.mat"
    return hsi_path, msi_path, sri_path

def main():
    for cd, dataset in enumerate(DATASETS, start=1):
        img_paths = glob.glob(f'data/GT/{dataset}/*.mat')
        for cm, method in enumerate(METHODS, start=1):
            for cs, scale in enumerate(SCALES, start=1):
                for ci, img_path in enumerate(img_paths, start=1):
                    img = Path(img_path).stem
                    hsi_path, msi_path, sri_path = get_paths(dataset, method, scale, img)
                    print(f"dataset: {dataset} ({cd}/{len(DATASETS)}), method: {method} ({cm}/{len(METHODS)}), scale: {scale} ({cs}/{len(SCALES)}), img: {img}({ci}/{len(img_paths)})")
                    if not os.path.exists(sri_path):
                        cmd = f'''hsi_path='{hsi_path}';msi_path='{msi_path}';sri_path='{sri_path}';{method}_run'''
                        matlabcmd(cmd)

if __name__ == "__main__":
    main()
