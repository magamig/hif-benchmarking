<h1 align="center"> 🌇🌆 Hyperspectral Image Fusion Benchmarking 🏙🌃 </h1>

**Comparison of the multispectral (MS) and hyperspectral (HS) image fusion techniques used for the spatial resolution enhancement of HS images.**

![diagram](aux/diagram.png)

Existing hyperspectral imaging systems produce images that lack spatial resolution due to hardware limitations. Even with the proven efficacy of this technology in several computer vision tasks, the aforementioned limitation obstructs its applicability. Contrarily, conventional RGB images have a much larger resolution with just three spectra. Since the issue of low resolution images arises from hardware limitations, there have been several developments in software-based approaches to improve the spatial resolution of hyperspectral images.

This work allows for an **easy-to-use framework for testing and comparing existing hyperspectral image fusion (HIF) methods** for spatial resolution enhancement.

## Content

* [Instructions](#instructions)
* [Datasets](#datasets)
* [Methods](#methods)
    * [Implemented Methods](#implemented-methods)
    * [Other Methods](#other-methods)
    * [Extensions](#extensions)
* [Metrics](#metrics)
* [Requirements](#requirements)


## Instructions

Download and process dataset(s) (e.g.: CAVE). This will also create MS image and downsampled HS image by a factor of 4 (or any other power of 2):

```
python main/dataset_CAVE.py 4
```

Run all algorithms over the datasets (you can edit ``run.py`` to customize the combinatory that you wish to process in terms of datasets, methods and scaling factors):

```
python main/run.py
```

Finally, compute the metrics that compare the output of the image fusion methods with the ground truth data:

```
python main/metrics.py
```

## Datasets

| Method | Year | Resolution | Download | Paper |
| --- | --- | --- | --- | --- |
| [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/) | 2008 | 512x512x31 | [URL](https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip) | [Yasuma, F., Mitsunaga, T., Iso, D., & Nayar, S. K. (2010). Generalized assorted pixel camera: postcapture control of resolution, dynamic range, and spectrum. IEEE transactions on image processing, 19(9), 2241-2253.](http://www1.cs.columbia.edu/CAVE/publications/pdfs/Yasuma_TR08.pdf) |
| [Harvard](http://vision.seas.harvard.edu/hyperspec/index.html) | 2011 | 1040x1392x31 | [URL](http://vision.seas.harvard.edu/hyperspec/d2x5g3/) | [Chakrabarti, A., & Zickler, T. (2011, June). Statistics of real-world hyperspectral images. In CVPR 2011 (pp. 193-200). IEEE.](http://vision.seas.harvard.edu/hyperspec/CZ_hss.pdf) |

## Methods

Hyperspectral image fusion (HIF) methods with code publicly available.

### Implemented Methods

Methods with code available together with an implemented wrapper in this repository (some of the wrappers are adapted from "Hyperspectral and Multispectral Data Fusion: A Comparative Review" [^1]).

| Method | Year | Code | Paper |
| --- | --- | --- | --- |
| SFIM* | 2000 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Liu, J. G. (2000). Smoothing filter-based intensity modulation: A spectral preserve image fusion technique for improving spatial details. International Journal of Remote Sensing, 21(18), 3461-3472.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.2091&rep=rep1&type=pdf) |
| MAPSMM | 2004 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Eismann, M. T. (2004). Resolution enhancement of hyperspectral imagery using maximum a posteriori estimation with a stochastic mixing model. University of Dayton.](https://www.proquest.com/openview/4c48da6b5ba634f91349241a57d830d4/) |
| GLP* | 2006 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Aiazzi, B., Alparone, L., Baronti, S., Garzelli, A., & Selva, M. (2006). MTF-tailored multiscale fusion of high-resolution MS and Pan imagery. Photogrammetric Engineering & Remote Sensing, 72(5), 591-596.](https://www.ingentaconnect.com/contentone/asprs/pers/2006/00000072/00000005/art00007?crawler=true&mimetype=application/pdf) |
| GSA | 2007 | [Matlab](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip) | [Aiazzi, B., Baronti, S., & Selva, M. (2007). Improving component substitution pansharpening through multivariate regression of MS +Pan data. IEEE Transactions on Geoscience and Remote Sensing, 45(10), 3230-3239.](https://d1wqtxts1xzle7.cloudfront.net/48446856/tgrs.2007.90100720160830-4045-b5r3a4-with-cover-page-v2.pdf?Expires=1650037886&Signature=d8gad3UNRLz-JrHo~fsLTSMVaaTKtKzsxHTi1GPlvO4BoVpiIIoRldM7JHyqJXozN7aEIIj-mC3wflIkODFGkULcrJhQ-v1X-pCmAAEByW5aDxftC8RB7X7kCdIHwfx~xfhfE0YkKuzaJOw2ZGFem6KUFX~DNts2CZNN524oEaAXzZeGm~TpK6eZnEPPFRamiREXzyg4~QfoAw~TFuRD8uLbQ9BSCEkpvWblDnFdsgseVseF4AJ5J4HFzK3yuBTtDgQgDwLG29yJg-ViccakE~zMau7eoDFZPs594MOrOziuUXJGumeg4MWeqidO7EXaiylVQs0u5yfa~Cwo1ZZvaw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) |
| CNMF | 2011 | [Python](https://naotoyokoya.com/assets/zip/CNMF_Python.zip) [Matlab](https://naotoyokoya.com/assets/zip/CNMF_MATLAB.zip) | [Yokoya, N., Yairi, T., & Iwasaki, A. (2011, July). Coupled non-negative matrix factorization (CNMF) for hyperspectral and multispectral data fusion: Application to pasture classification. In 2011 IEEE International Geoscience and Remote Sensing Symposium (pp. 1779-1782). IEEE.](http://www.naotoyokoya.com/assets/pdf/NYokoyaTGRS2012.pdf) |
| GSOMP | 2014 | [Matlab](http://staffhome.ecm.uwa.edu.au/~00053650/code.html) | [Akhtar, N., Shafait, F., & Mian, A. (2014, September). Sparse spatio-spectral representation for hyperspectral image super-resolution. In European conference on computer vision (pp. 63-78). Springer, Cham.](https://link.springer.com/content/pdf/10.1007/978-3-319-10584-0_5.pdf) |
| HySure | 2014 | [Matlab](https://github.com/alfaiate/HySure) | [Simoes, M., Bioucas-Dias, J., Almeida, L. B., & Chanussot, J. (2014, October). Hyperspectral image superresolution: An edge-preserving convex formulation.Hysure In 2014 IEEE International Conference on Image Processing (ICIP) (pp. 4166-4170). IEEE.](http://www.lx.it.pt/~bioucas/files/icip_2014_hs_sr_convex.pdf) |
| BayesianSparse (very slow) | 2015 | Matlab | [Akhtar, N., Shafait, F., & Mian, A. (2015). Bayesian sparse representation for hyperspectral image super resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3631-3640).](https://openaccess.thecvf.com/content_cvpr_2015/papers/Akhtar_Bayesian_Sparse_Representation_2015_CVPR_paper.pdf) |
| FUSE | 2015 | [Matlab](http://wei.perso.enseeiht.fr/demo/MCMCFusion.7z) | [Wei, Q., Dobigeon, N., & Tourneret, J. Y. (2015). Bayesian fusion of multi-band images. IEEE Journal of Selected Topics in Signal Processing, 9(6), 1117-1127.](http://wei.perso.enseeiht.fr/papers/WEI_JSTSP_final.pdf) |
| SupResPALM | 2015 | [Matlab](https://github.com/lanha/SupResPALM) | [Lanaras, C., Baltsavias, E., & Schindler, K. (2015). Hyperspectral super-resolution by coupled spectral unmixing. In Proceedings of the IEEE international conference on computer vision (pp. 3586-3594).](https://openaccess.thecvf.com/content_iccv_2015/papers/Lanaras_Hyperspectral_Super-Resolution_by_ICCV_2015_paper.pdf) |
| [NSSR](https://see.xidian.edu.cn/faculty/wsdong/HSI_SR_Project.htm) | 2016 | [Matlab](https://see.xidian.edu.cn/faculty/wsdong/Code_release/NSSR_HSI_SR.rar) | [Dong, W., Fu, F., Shi, G., Cao, X., Wu, J., Li, G., & Li, X. (2016). Hyperspectral image super-resolution via non-negative structured sparse representation. IEEE Transactions on Image Processing, 25(5), 2337-2352.](https://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/NSSR_HSI_TIP16.pdf) |
| CMS | 2018 | [Matlab](https://drive.google.com/open?id=1AptXqCiTgxbyWPRg7g5fiDZ6KWV-qsKJ) | [Zhang, L., Wei, W., Bai, C., Gao, Y., & Zhang, Y. (2018). Exploiting clustering manifold structure for hyperspectral imagery super-resolution. IEEE Transactions on Image Processing, 27(12), 5969-5982.](https://ieeexplore.ieee.org/abstract/document/8424415)
| CNN-FUS | 2018 | [Matlab](https://github.com/renweidian/CNN-FUS) | [Dian, R., Li, S., & Kang, X. (2020). Regularizing hyperspectral and multispectral image fusion by CNN denoiser. IEEE transactions on neural networks and learning systems, 32(3), 1124-1135.](https://github.com/renweidian/CNN-FUS/blob/master/TNNLS-2020.pdf?raw=true)
| CSTF (unstable) | 2018 | [Matlab](https://drive.google.com/open?id=12eleEjv7wKQxFCBUcIGkEl-wiUiJxwTv) | [Li, S., Dian, R., Fang, L., & Bioucas-Dias, J. M. (2018). Fusing hyperspectral and multispectral images via coupled sparse tensor factorization. IEEE Transactions on Image Processing, 27(8), 4118-4130.](http://www.lx.it.pt/~bioucas/files/ieee_tip_fusion_sparse_tf.pdf)
| LTMR | 2019 | [Matlab](https://github.com/renweidian/LTMR) | [Dian, R., & Li, S. (2019). Hyperspectral image super-resolution via subspace-based low tensor multi-rank regularization. IEEE Transactions on Image Processing, 28(10), 5135-5146.](https://github.com/renweidian/LTMR/raw/master/TIP-2019.pdf) |
| LTTR | 2019 | [Matlab](https://github.com/renweidian/LTTR) | [Dian, R., Li, S., & Fang, L. (2019). Learning a low tensor-train rank representation for hyperspectral image super-resolution. IEEE transactions on neural networks and learning systems, 30(9), 2672-2683.](https://www.leyuanfang.com/wp-content/uploads/2022/02/2019-10-10.1109@TNNLS.2018.2885616.pdf) |

\* pan-sharpening methods adapted to HS–MS fusion [^1] via hypersharpening [^2].

### Other Methods

Code is available but wrapper is not implemented yet.

| Method | Year | Code | Paper |
| --- | --- | --- | --- |
| [MF](https://nae-lab.org/~rei/research/hh/) | 2011 | [Matlab](https://nae-lab.org/~rei/research/hh/matrix_factorization_hyperspectral_frozen.zip) | [Kawakami, R., Matsushita, Y., Wright, J., Ben-Ezra, M., Tai, Y. W., & Ikeuchi, K. (2011, June). High-resolution hyperspectral imaging via matrix factorization. In CVPR 2011 (pp. 2329-2336). IEEE.](https://nae-lab.org/~rei/research/hh/cvpr11/rei_cvpr.pdf) |
| SNMF | 2013 | [Matlab](https://mx.nthu.edu.tw/~tsunghan/download/SNNMF.rar) | [Wycoff, E., Chan, T. H., Jia, K., Ma, W. K., & Ma, Y. (2013, May). A non-negative sparse promoting algorithm for high resolution hyperspectral imaging. In 2013 IEEE International Conference on Acoustics, Speech and Signal Processing (pp. 1409-1413). IEEE.](http://people.eecs.berkeley.edu/~yima/psfile/ICASSP13-Hyperspectral_Submission_Copy.pdf) |
| BSR | 2015 | [Matlab](http://wei.perso.enseeiht.fr/demo/SparseFusion_2014-12-03.zip) | [Wei, Q., Bioucas-Dias, J., Dobigeon, N., & Tourneret, J. Y. (2015). Hyperspectral and multispectral image fusion based on a sparse representation. IEEE Transactions on Geoscience and Remote Sensing, 53(7), 3658-3668.](https://arxiv.org/pdf/1409.5729.pdf) |
| BlindFuse | 2016 | [Matlab](https://github.com/qw245/BlindFuse) | [Wei, Q., Bioucas-Dias, J., Dobigeon, N., Tourneret, J. Y., & Godsill, S. (2016, September). Blind model-based fusion of multi-band and panchromatic images. In 2016 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI) (pp. 21-25). IEEE.](https://hal.archives-ouvertes.fr/hal-01682975/document) |
| FUMI | 2016 | [Matlab](https://github.com/qw245/FUMI) | [Wei, Q., Bioucas-Dias, J., Dobigeon, N., Tourneret, J. Y., Chen, M., & Godsill, S. (2016). Multiband image fusion based on spectral unmixing. IEEE Transactions on Geoscience and Remote Sensing, 54(12), 7236-7249.](https://arxiv.org/pdf/1603.08720.pdf) |
| BRS | 2018 | [Matlab](https://github.com/mehrhardt/blind_remote_sensing) | [Bungert, L., Coomes, D. A., Ehrhardt, M. J., Rasch, J., Reisenhofer, R., & Schönlieb, C. B. (2018). Blind image fusion for hyperspectral imaging with the directional total variation. Inverse Problems, 34(4), 044003.](https://iopscience.iop.org/article/10.1088/1361-6420/aaaf63/pdf)
| DHSIS | 2018 | [Python](https://github.com/renweidian/DHSIS) | [Dian, R., Li, S., Guo, A., & Fang, L. (2018). Deep hyperspectral image sharpening. IEEE transactions on neural networks and learning systems, 29(11), 5345-5355.](https://www.leyuanfang.com/wp-content/uploads/2022/02/2018-13-dian2018.pdf)
| uSDN | 2018 | [Python](https://github.com/aicip/uSDN) | [Qu, Y., Qi, H., & Kwan, C. (2018). Unsupervised sparse dirichlet-net for hyperspectral image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2511-2520).](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.pdf) |
| DBIN | 2019 | [Python](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion) | [Wang, W., Zeng, W., Huang, Y., Ding, X., & Paisley, J. (2019). Deep blind hyperspectral image fusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4150-4159).](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf)
| CUCaNet | 2020 | [Python](https://github.com/danfenghong/ECCV2020_CUCaNet) |  [Yao, J., Hong, D., Chanussot, J., Meng, D., Zhu, X., & Xu, Z. (2020, August). Cross-attention in coupled unmixing nets for unsupervised hyperspectral super-resolution. In European Conference on Computer Vision (pp. 208-224). Springer, Cham.](https://arxiv.org/pdf/2007.05230.pdf) |
| GDD | 2020 | [Python](https://github.com/tuezato/guided-deep-decoder) |  [Uezato, T., Hong, D., Yokoya, N., & He, W. (2020, August). Guided deep decoder: Unsupervised image pair fusion. In European Conference on Computer Vision (pp. 87-102). Springer, Cham.](https://arxiv.org/pdf/2007.11766.pdf) |
| MHF-net | 2020 | [Python](https://github.com/XieQi2015/MHF-net) |  [Xie, Q., Zhou, M., Zhao, Q., Xu, Z., & Meng, D. (2020). MHF-net: An interpretable deep network for multispectral and hyperspectral image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence.](https://ieeexplore.ieee.org/document/9165231) |
| PZRes-Net | 2020 | [Python](https://github.com/zbzhzhy/PZRes-Net) | [Zhu, Z., Hou, J., Chen, J., Zeng, H., & Zhou, J. (2020). Hyperspectral image super-resolution via deep progressive zero-centric residual learning. IEEE Transactions on Image Processing, 30, 1423-1438.](https://arxiv.org/pdf/2006.10300.pdf) |
| RecHSISR | 2020 | [Python](https://github.com/JiangtaoNie/Rec_HSISR_PixAwaRefin) |  [Wei, W., Nie, J., Zhang, L., & Zhang, Y. (2020). Unsupervised recurrent hyperspectral imagery super-resolution using pixel-aware refinement. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-15.](https://ieeexplore.ieee.org/abstract/document/9292466/) |
| SSRNET | 2020 | [Python](https://github.com/hw2hwei/SSRNET) | [Zhang, X., Huang, W., Wang, Q., & Li, X. (2020). SSR-NET: Spatial–spectral reconstruction network for hyperspectral and multispectral image fusion. IEEE Transactions on Geoscience and Remote Sensing, 59(7), 5953-5965.](https://ieeexplore.ieee.org/abstract/document/9186332/) |
| TONWMD | 2020 | [Python](https://github.com/liuofficial/TONWMD) | [Shen, D., Liu, J., Xiao, Z., Yang, J., & Xiao, L. (2020). A twice optimizing net with matrix decomposition for hyperspectral and multispectral image fusion. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 4095-4110.](https://ieeexplore.ieee.org/abstract/document/9141409) |
| Two-CNN | 2020 | [Matlab](https://github.com/polwork/Hyperspectral-and-Multispectral-fusion-via-Two-branch-CNN) | [Yang, J., Zhao, Y. Q., & Chan, J. C. W. (2018). Hyperspectral and multispectral image fusion via deep two-branches convolutional neural network. Remote Sensing, 10(5), 800.](https://www.mdpi.com/2072-4292/10/5/800/htm) |
| UAL | 2020 | [Python](https://github.com/JiangtaoNie/UAL-CVPR2020) |  [Zhang, L., Nie, J., Wei, W., Zhang, Y., Liao, S., & Shao, L. (2020). Unsupervised adaptation learning for hyperspectral imagery super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3073-3082).](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.pdf) |
| ADMM-HFNET | 2021 | [Python](https://github.com/liuofficial/ADMM-HFNet) | [Shen, D., Liu, J., Wu, Z., Yang, J., & Xiao, L. (2021). ADMM-HFNet: A Matrix Decomposition-Based Deep Approach for Hyperspectral Image Fusion. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-17.](https://ieeexplore.ieee.org/abstract/document/9547385) |
| Fusformer | 2021 | [Python](https://github.com/J-FHu/Fusformer) |  [Hu, J. F., Huang, T. Z., & Deng, L. J. (2021). Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution. arXiv preprint arXiv:2109.02079.](https://arxiv.org/pdf/2109.02079.pdf) |
| [MoG-DCN](https://see.xidian.edu.cn/faculty/wsdong/Projects/MoG-DCN.htm) | 2021 | [Python](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution) | [Dong, W., Zhou, C., Wu, F., Wu, J., Shi, G., & Li, X. (2021). Model-guided deep hyperspectral image super-resolution. IEEE Transactions on Image Processing, 30, 5754-5768.](https://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/HSISR-TIP-v4.pdf) |
| HyperFusion | 2021 | [Python](https://github.com/saber-zero/HyperFusion) |  [Tian, X., Zhang, W., Chen, Y., Wang, Z., & Ma, J. (2021). Hyperfusion: A computational approach for hyperspectral, multispectral, and panchromatic image fusion. IEEE Transactions on Geoscience and Remote Sensing.](https://ieeexplore.ieee.org/abstract/document/9615043) |
| [HSRnet](https://liangjiandeng.github.io/Projects_Res/HSRnet_2021tnnls.html) | 2021 | [Python](https://github.com/liangjiandeng/HSRnet) | [Dong, W., Zhou, C., Wu, F., Wu, J., Shi, G., & Li, X. (2021). Model-guided deep hyperspectral image super-resolution. IEEE Transactions on Image Processing, 30, 5754-5768.](https://liangjiandeng.github.io/papers/2021/HSRnet_tnnls_2021.pdf) |
| TSFN | 2021 | [Python](https://github.com/xiuheng-wang/Sylvester_TSFN_MDC_HSI_superresolution) | [Wang, X., Chen, J., Wei, Q., & Richard, C. (2021). Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation. IEEE Transactions on Circuits and Systems for Video Technology.](https://arxiv.org/pdf/2009.04237.pdf) |
| u2MDN | 2021 | [Python](https://github.com/yingutk/u2MDN) |  [Qu, Y., Qi, H., Kwan, C., Yokoya, N., & Chanussot, J. (2021). Unsupervised and unregistered hyperspectral image super-resolution with mutual Dirichlet-Net. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-18.](https://arxiv.org/pdf/1904.12175.pdf) |
| DBSR | 2022 | [Python](https://github.com/JiangtaoNie/DBSR) |  [Zhang, L., Nie, J., Wei, W., Li, Y., & Zhang, Y. (2020). Deep blind hyperspectral image super-resolution. IEEE Transactions on Neural Networks and Learning Systems, 32(6), 2388-2400.](https://ieeexplore.ieee.org/abstract/document/9136736) |
| [DHIF](https://see.xidian.edu.cn/faculty/wsdong/Projects/TCI2022-DHIF-Net/DHIF-Net.htm) | 2022 | [Python](https://github.com/TaoHuang95/DHIF-Net) | [Huang, T., Dong, W., Wu, J., Li, L., Li, X., & Shi, G. (2022). Deep Hyperspectral Image Fusion Network With Iterative Spatio-Spectral Regularization. IEEE Transactions on Computational Imaging, 8, 201-214.](https://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/TCI-2022-Deep_Hyperspectral_Image_Fusion.pdf) |
| HSI-CSR | 2022 | [Caffe](https://github.com/ColinTaoZhang/HSI-SR) |  [Fu, Y., Zhang, T., Zheng, Y., Zhang, D., & Huang, H. (2019). Hyperspectral image super-resolution with optimized RGB guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11661-11670).](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Hyperspectral_Image_Super-Resolution_With_Optimized_RGB_Guidance_CVPR_2019_paper.pdf) |
| HSISR | 2022 | [Python](https://github.com/kli8996/HSISR) | [Li, K., Dai, D., & Van Gool, L. (2022). Hyperspectral Image Super-Resolution with RGB Image Super-Resolution as an Auxiliary Task. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 3193-3202).](https://openaccess.thecvf.com/content/WACV2022/papers/Li_Hyperspectral_Image_Super-Resolution_With_RGB_Image_Super-Resolution_as_an_Auxiliary_WACV_2022_paper.pdf) |
| MIAE | 2022 | [Python](https://github.com/liuofficial/MIAE/) |  [Liu, J., Wu, Z., Xiao, L., & Wu, X. J. (2022). Model Inspired Autoencoder for Unsupervised Hyperspectral Image Super-Resolution. IEEE Transactions on Geoscience and Remote Sensing.](https://arxiv.org/pdf/2110.11591.pdf) |
| NonRegSRNet | 2022 | [Python](https://github.com/saber-zero/NonRegSRNet) |  [Zheng, K., Gao, L., Hong, D., Zhang, B., & Chanussot, J. (2021). NonRegSRNet: A Nonrigid Registration Hyperspectral Super-Resolution Network. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-16.](https://github.com/saber-zero/NonRegSRNet/blob/main/NonRegSRNet__a_Non_rigid_Registration_Hyperspectral_Super_Resolution_Network.pdf?raw=true) |
| RAFnet | 2022 | [Python](https://github.com/RuiyingLu/RAFnet) |  [Lu, R., Chen, B., Cheng, Z., & Wang, P. (2020). RAFnet: Recurrent attention fusion network of hyperspectral and multispectral images. Signal Processing, 177, 107737.](https://www.sciencedirect.com/science/article/pii/S0165168420302802) |
| SpfNet | 2022 | [Python](https://github.com/liuofficial/SpfNet) |  [Liu, J., Shen, D., Wu, Z., Xiao, L., Sun, J., & Yan, H. (2022). Patch-Aware Deep Hyperspectral and Multispectral Image Fusion by Unfolding Subspace-Based Optimization Model. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.](https://ieeexplore.ieee.org/abstract/document/9670687) |
| UDALN | 2022 | [Python](https://github.com/JiaxinLiCAS/UDALN_GRSL) |  [Li, J., Zheng, K., Yao, J., Gao, L., & Hong, D. (2022). Deep Unsupervised Blind Hyperspectral and Multispectral Data Fusion. IEEE Geoscience and Remote Sensing Letters, 19, 1-5.](https://www.researchgate.net/profile/Jiaxin-Li-97/publication/358719445_Deep_Unsupervised_Blind_Hyperspectral_and_Multispectral_Data_Fusion/links/621068166c472329dcf44a53/Deep-Unsupervised-Blind-Hyperspectral-and-Multispectral-Data-Fusion.pdf) |

### Extensions

HSI methods' extensions with code publicly available. These should be regarded as extensions to the base pipelines and not as a new methods. The extensions take a super-resolution image (output of the HSI method) together with the MS and HS images as inputs, and provide an improved super-resolution image as output. The wrappers for these extensions are not implemented in this repository yet.

| Method | Year | Code | Paper |
| --- | --- | --- | --- |
| TVTVHS | 2021 | [Python](https://github.com/marijavella/hs-sr-tvtv) | [Vella, M., Zhang, B., Chen, W., & Mota, J. F. (2021, September). Enhanced Hyperspectral Image Super-Resolution via RGB Fusion and TV-TV Minimization. In 2021 IEEE International Conference on Image Processing (ICIP) (pp. 3837-3841). IEEE.](https://arxiv.org/pdf/2106.07066.pdf) |
| DeepGrad | 2022 | [Matlab](https://github.com/xiuheng-wang/Deep_gradient_HSI_superresolution) | [Wang, X., Chen, J., & Richard, C. (2022). Hyperspectral Image Super-resolution with Deep Priors and Degradation Model Inversion. arXiv preprint arXiv:2201.09851.](https://arxiv.org/pdf/2201.09851.pdf) |

## Metrics

To evaluate the quality of the methods, the output of the superresolution methods is compared with the ground truth of the dataset. We compute several metrics (listed below) using [sewar](https://github.com/andrewekhalel/sewar).

| Acronym | Full Name | Paper |
| --- | --- | --- |
| RMSE | Root Mean Squared Error | - |
| PSNR | Peak Signal-to-Noise Ratio | [Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) |
| SSIM | Structural Similarity Index | [Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) |
| UQI | Universal Quality Image Index | [Wang, Z., & Bovik, A. C. (2002). A universal image quality index. IEEE signal processing letters, 9(3), 81-84.](https://ieeexplore.ieee.org/abstract/document/995823) |
| MS-SSIM | Multi-scale Structural Similarity Index | [Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003, November). Multiscale structural similarity for image quality assessment. In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.](https://utw10503.utweb.utexas.edu/publications/2003/zw_asil2003_msssim.pdf) |
| ERGAS | Erreur Relative Globale Adimensionnelle de Synthèse | [Wald, L. (2000, January). Quality of high resolution synthesised images: Is there a simple criterion?. In Third conference" Fusion of Earth data: merging point measurements, raster maps and remotely sensed images" (pp. 99-103). SEE/URISCA.](https://hal.archives-ouvertes.fr/hal-00395027/file/ergas_-_wald_2000.pdf) |
| SCC | Spatial Correlation Coefficient | [Zhou, J., Civco, D. L., & Silander, J. A. (1998). A wavelet transform method to merge Landsat TM and SPOT panchromatic data. International journal of remote sensing, 19(4), 743-757.](https://www.tandfonline.com/doi/abs/10.1080/014311698215973) |
| RASE | Relative Average Spectral Error | [González-Audícana, M., Saleta, J. L., Catalán, R. G., & García, R. (2004). Fusion of multispectral and panchromatic images using improved IHS and PCA mergers based on wavelet decomposition. IEEE Transactions on Geoscience and Remote sensing, 42(6), 1291-1299.](http://aet.org.es/rnta/reunion2008/Jornadas-fusion-imagenes/articulos/Fusion%20of%20multi%20and%20pan%20image%20using%20IHS%20and%20PCA%20based%20on%20Wav.pdf) |
| SAM | Spectral Angle Mapper | [Yuhas, R. H., Goetz, A. F., & Boardman, J. W. (1992, June). Discrimination among semi-arid landscape endmembers using the spectral angle mapper (SAM) algorithm. In JPL, Summaries of the Third Annual JPL Airborne Geoscience Workshop. Volume 1: AVIRIS Workshop.](https://ntrs.nasa.gov/api/citations/19940012238/downloads/19940012238.pdf) |
| VIF | Visual Information Fidelity | [Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual quality. IEEE Transactions on image processing, 15(2), 430-444.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.477.2659&rep=rep1&type=pdf) |
| PSNR-B | Block Sensitive - Peak Signal-to-Noise Ratio | [Yim, C., & Bovik, A. C. (2010). Quality assessment of deblocked images. IEEE Transactions on Image Processing, 20(1), 88-98.](http://www.kresttechnology.com/krest-academic-projects/krest-mtech-projects/ECE/dspmt/[50].pdf) |
| Q2ⁿ * | Q2ⁿ | [Garzelli, A., & Nencini, F. (2009). Hypercomplex quality assessment of multi/hyperspectral images. IEEE Geoscience and Remote Sensing Letters, 6(4), 662-665.](https://www.researchgate.net/profile/Andrea-Garzelli/publication/224560382_Hypercomplex_Quality_Assessment_of_MultiHyperspectral_Images/links/0f317538da04c396e5000000/Hypercomplex-Quality-Assessment-of-Multi-Hyperspectral-Images.pdf)

\* to be implemented in the future.

## Requirements

- `pip install -r aux/requirements.txt`
- [SPAMS-2.6](http://thoth.inrialpes.fr/people/mairal/spams/downloads.html)
- [MatConvNet](https://www.vlfeat.org/matconvnet/install/)

[^1]: Yokoya, N., Grohnfeldt, C., & Chanussot, J. (2017). Hyperspectral and multispectral data fusion: A comparative review of the recent literature. IEEE Geoscience and Remote Sensing Magazine, 5(2), 29-56. [[paper]](https://naotoyokoya.com/assets/pdf/NYokoyaGRSM2017.pdf) [[code]](https://openremotesensing.net/wp-content/uploads/2017/11/HSMSFusionToolbox.zip)

[^2]: Selva, M., Aiazzi, B., Butera, F., Chiarantini, L., & Baronti, S. (2015). Hyper-sharpening: A first approach on SIM-GA data. IEEE Journal of selected topics in applied earth observations and remote sensing, 8(6), 3008-3024. [[paper]](https://ieeexplore.ieee.org/abstract/document/7134741)