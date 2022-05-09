This code provides a demo code of the following paper. 
'Guided Deep Decoder: Unsupervised Image Pair Fusion', Uezato et al., ECCV 2020.
Paper is available: https://arxiv.org/abs/2007.11766

The code uses some functions from the following paper.
'Deep Image Prior', Ulyanov et al., CVPR 2018.

There are three demo codes.

1. GDD_demo_HS - code of hyperspectral and multispectral image fusion.
2. GDD_demo_PAN - code of panchromatic and multispectral image fusion.
3. GDD_demo_denoising - code of flash and no-flash image fusion.

When you apply the method to a different dataset, you need to change the spectral respnse function 'R' and define a new downsampling function.

Please find the paper for more detailed information.
If you find any bug or have any suggestion, please contact Tatsumi Uezato (RIKEN AIP).
