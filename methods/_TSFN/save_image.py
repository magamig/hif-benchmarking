import numpy as np
import tifffile as tiff
import os
import scipy.io as scio

def save_image(output, label, filename, out_results_path):
    if not os.path.exists(out_results_path):
        os.makedirs(out_results_path)
    image = output
    image = np.clip(image, 0, 1) 
    image = image * 255
    tiff.imsave(out_results_path + str(filename) + '.tif', image.astype(np.uint8))
    
    output = output.transpose([1, 2, 0])
    output = np.clip(output, 0, 1)
    label = label.transpose([1, 2, 0])
    scio.savemat(out_results_path + str(filename) + '.mat', {'sr':output, 'gt':label})

