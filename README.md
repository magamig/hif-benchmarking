# Multispectral and Hyperspectral Image Fusion Benchmarking

Comparison the MS and HS Image Fusion techniques used for the spatial resolution enhancement of hyperspectral images.

## Instructions

Download and process dataset(s) (e.g.: CAVE):

```
bash aux/CAVE/download.sh
python aux/CAVE/process.py
```

Post process dataset(s) to create MS image and downsampled HS image by a factor of 4 (or any other power of 2):

```
python aux/postprocess.py CAVE 4
```

Run all algorithms over the dataset:

```
python run.py
```

You can edit ``run.py`` to customize the combinatory that you wish to process in terms of datasets, methods and scaling factors.

## Methods

| Method | Year | Code | Paper |
| --- | --- | --- | --- |
| CNMF | 2011 | [Python](https://naotoyokoya.com/assets/zip/CNMF_Python.zip) [Matlab](https://naotoyokoya.com/assets/zip/CNMF_MATLAB.zip) | [Yokoya, N., Yairi, T., & Iwasaki, A. (2011, July). Coupled non-negative matrix factorization (CNMF) for hyperspectral and multispectral data fusion: Application to pasture classification. In 2011 IEEE International Geoscience and Remote Sensing Symposium (pp. 1779-1782). IEEE.](http://www.naotoyokoya.com/assets/pdf/NYokoyaTGRS2012.pdf) |
| FUSE | 2015 | [Matlab](http://wei.perso.enseeiht.fr/demo/MCMCFusion.7z) | [Wei, Q., Dobigeon, N., & Tourneret, J. Y. (2015). Bayesian fusion of multi-band images. IEEE Journal of Selected Topics in Signal Processing, 9(6), 1117-1127.](http://wei.perso.enseeiht.fr/papers/WEI_JSTSP_final.pdf) |
| SFIM | 2000 | [Matlab](https://openremotesensing.net/kb/codes/) | [Liu, J. G. (2000). Smoothing filter-based intensity modulation: A spectral preserve image fusion technique for improving spatial details. International Journal of Remote Sensing, 21(18), 3461-3472.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.2091&rep=rep1&type=pdf) |
| HySure | 2014 | [Matlab](https://github.com/alfaiate/HySure) | [Simoes, M., Bioucas-Dias, J., Almeida, L. B., & Chanussot, J. (2014, October). Hyperspectral image superresolution: An edge-preserving convex formulation. In 2014 IEEE International Conference on Image Processing (ICIP) (pp. 4166-4170). IEEE.](http://www.lx.it.pt/~bioucas/files/icip_2014_hs_sr_convex.pdf) |
