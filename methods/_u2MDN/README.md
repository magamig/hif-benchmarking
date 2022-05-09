# u2MDN
This is a demo code for 'Unsupervised and Unregistered Hyperspectral Image Super-Resolution with Mutual Dirichlet-Net'. The code is for research purposes only. All rights reserved. 

Contact information 
Ying Qu: yqu3@vols.utk.edu

Requirements: 

python 3.7.6
tensorflow 1.13.1
scipy


To run the code, please run 'python pan_remote_demo.py'. 
'pbAuto_mi.py' is used to build the training models. 
'utils_remote.py' is used to load the data.

This method is an unsupervised hyperspectral image super-resolution method without image registration. The input of the network should be the low-resolution HSI and high-resolution MSI. The high-resolution HSI is only used to evaluate the reconstruction accuracy.  Note that the evaluation results may vary due to different network initialization. 


The default parameter setting is fine in our experiments. However, it may not be the optimal parameter group. Please adjust the parameters based on the input images. 

1. The default parameter is set for the remote sensing images with hundreds of spectral bands. For Harvard or CAVE dataset with fewer spectral bands, please reduce the 'num_hidden' and 'num_ly'. For example, num_hidden=15, num_ly=3. 

2. The parameter for sparsity is defined with "s_p", please adjust it according to the input image. If "s_p" is too large, it may not reconstruct the image well. If "s_p" is too small, it may not be effective. 

3. The parameter for mutual information is defined with "mi_p". Similar to the sparsity parameter, please adjust it according to the image. 


If you find the code helpful, please cite the following paper. 

Ying Qu, Hairong Qi and Chiman Kwan, Naoto Yokoya, Jocelyn Chanussot. “Unsupervised and Unregistered Hyperspectral Image Super-Resolution with Mutual Dirichlet-Net”, in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3079518.


@article{qu2021u2mdn,
  author={Qu, Ying and Qi, Hairong and Kwan, Chiman and Yokoya, Naoto and Chanussot, Jocelyn},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Unsupervised and Unregistered Hyperspectral Image Super-Resolution With Mutual Dirichlet-Net}, 
  year={2021},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2021.3079518}}
