# Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation

Steps:

Process the public dataset CAVE:

1. Run ./data/png2mat.m to convert PNG files to MAT files;

2. Run ./data/data_label.m to creating training/test data and generating HR-MSI;

3. Run ./data/mat2tif.m to convert MAT files to TIF files;

Train and test the two-stream fusion network TSFN:

1. Run train.py to train the TSFN; 

2. Run test.py to test the TSFN;

Run ./enhancement/enhance_adaptive.m to obtain the final HR-HSI estimation.

For any questions, feel free to email me at xiuheng.wang@mail.nwpu.edu.cn.

If you find this code helpful, please kindly cite:

@article{wang2021hyperspectral,
  title={Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation},
  author={Wang, Xiuheng and Chen, Jie and Wei, Qi and Richard, C{\'e}dric},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2021},
  publisher={IEEE}
}
