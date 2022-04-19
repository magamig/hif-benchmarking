# Multispectral and Hyperspectral Image Fusion Benchmarking

**Comparison the MS and HS Image Fusion techniques used for the spatial resolution enhancement of hyperspectral images.**

Existing hyperspectral imaging systems produce images that lack spatial resolution due to hardware limitations. Even with the proven efficacy of this technology in several computer vision tasks, the aforementioned limitation obstructs its applications. Contrarily, conventional RGB images have a much larger resolution with just three spectra. Since the issue of low resolution images arises from hardware limitations, there have been several developments in software-based approaches to improve the resolution of hyperspectral images.

![diagram](diagram.png)

This work allows for an easy-to-use framework for testing and comparing existing image fusion methods of resolution enhancement for hyperspectral images. Part of the code is adapted from "Hyperspectral and Multispectral Data Fusion: A Comparative Review" [^1].

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
| SFIM | 2000 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Liu, J. G. (2000). Smoothing filter-based intensity modulation: A spectral preserve image fusion technique for improving spatial details. International Journal of Remote Sensing, 21(18), 3461-3472.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.2091&rep=rep1&type=pdf) |
| HySure | 2014 | [Matlab](https://github.com/alfaiate/HySure) | [Simoes, M., Bioucas-Dias, J., Almeida, L. B., & Chanussot, J. (2014, October). Hyperspectral image superresolution: An edge-preserving convex formulation. In 2014 IEEE International Conference on Image Processing (ICIP) (pp. 4166-4170). IEEE.](http://www.lx.it.pt/~bioucas/files/icip_2014_hs_sr_convex.pdf) |
| GLP | 2006 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Aiazzi, B., Alparone, L., Baronti, S., Garzelli, A., & Selva, M. (2006). MTF-tailored multiscale fusion of high-resolution MS and Pan imagery. Photogrammetric Engineering & Remote Sensing, 72(5), 591-596.](https://www.ingentaconnect.com/contentone/asprs/pers/2006/00000072/00000005/art00007?crawler=true&mimetype=application/pdf) |
| GSA | 2007 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Aiazzi, B., Baronti, S., & Selva, M. (2007). Improving component substitution pansharpening through multivariate regression of MS +Pan data. IEEE Transactions on Geoscience and Remote Sensing, 45(10), 3230-3239.](https://d1wqtxts1xzle7.cloudfront.net/48446856/tgrs.2007.90100720160830-4045-b5r3a4-with-cover-page-v2.pdf?Expires=1650037886&Signature=d8gad3UNRLz-JrHo~fsLTSMVaaTKtKzsxHTi1GPlvO4BoVpiIIoRldM7JHyqJXozN7aEIIj-mC3wflIkODFGkULcrJhQ-v1X-pCmAAEByW5aDxftC8RB7X7kCdIHwfx~xfhfE0YkKuzaJOw2ZGFem6KUFX~DNts2CZNN524oEaAXzZeGm~TpK6eZnEPPFRamiREXzyg4~QfoAw~TFuRD8uLbQ9BSCEkpvWblDnFdsgseVseF4AJ5J4HFzK3yuBTtDgQgDwLG29yJg-ViccakE~zMau7eoDFZPs594MOrOziuUXJGumeg4MWeqidO7EXaiylVQs0u5yfa~Cwo1ZZvaw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) |
| SupResPALM | 2015 | [Matlab](https://github.com/lanha/SupResPALM) | [Lanaras, C., Baltsavias, E., & Schindler, K. (2015). Hyperspectral super-resolution by coupled spectral unmixing. In Proceedings of the IEEE international conference on computer vision (pp. 3586-3594).](https://openaccess.thecvf.com/content_iccv_2015/papers/Lanaras_Hyperspectral_Super-Resolution_by_ICCV_2015_paper.pdf) |
| MAPSMM | 2004 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Eismann, M. T. (2004). Resolution enhancement of hyperspectral imagery using maximum a posteriori estimation with a stochastic mixing model. University of Dayton.](https://www.proquest.com/openview/4c48da6b5ba634f91349241a57d830d4/) |
| NSSR | 2016 | [Matlab](https://see.xidian.edu.cn/faculty/wsdong/Code_release/NSSR_HSI_SR.rar) | [Dong, W., Fu, F., Shi, G., Cao, X., Wu, J., Li, G., & Li, X. (2016). Hyperspectral image super-resolution via non-negative structured sparse representation. IEEE Transactions on Image Processing, 25(5), 2337-2352.](https://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/NSSR_HSI_TIP16.pdf) |

*Pan-sharpening methods adapted to HS–MS fusion via hypersharpening. [^1]


[^1]: Yokoya, N., Grohnfeldt, C., & Chanussot, J. (2017). Hyperspectral and multispectral data fusion: A comparative review of the recent literature. IEEE Geoscience and Remote Sensing Magazine, 5(2), 29-56. [[paper]](https://naotoyokoya.com/assets/pdf/NYokoyaGRSM2017.pdf) [[code]](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip)