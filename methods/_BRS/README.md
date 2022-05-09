# Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation
This MATLAB code allows to reproduce the results of <b>Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation</b> [1].

[1] Bungert, L., Coomes, D. A., Ehrhardt, M. J., Rasch, J., Reisenhofer, R., & Schönlieb, C.-B. (2018). Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation. Inverse Problems, 34(4), 044003. https://doi.org/10.1088/1361-6420/aaaf63 http://arxiv.org/abs/1710.05705

The aim of [1] is to fuse a hyperspectral image of low spatial resolution with a photograph of higher spatial resolution. Three examples are shown below. The examples on the left and middle where acquired from a plane flying over Spain to study vegetation. The example on the right has been acquired from a satellite.

<p align="center">
<img src="https://github.com/mehrhardt/blind_remote_sensing/blob/master/fig_1a.png" width="20%" border="0"/>
<img src="https://github.com/mehrhardt/blind_remote_sensing/blob/master/fig_1b.png" width="20%" border="0"/>
<img src="https://github.com/mehrhardt/blind_remote_sensing/blob/master/fig_1c.png" width="20%" border="0"/></p>

The higher spatial resolution photograph is very important to resolve fine details. The example below compares the proposed regularizer "directional total variation" (dTV) to a more standard regularizer "total variation" (TV).
<p align="center"><img src="https://github.com/mehrhardt/blind_remote_sensing/blob/master/fig_3.png" width="60%" border="0"/></p>

The mathematical model usually assumes that the hyperspectral image and the high-resolution photograph are perfectly aligned. For real data this is rarely the case. The proposed model estimates and corrects for a possible mismatch during the reconstruction. The example below shows the impact of the proposed "blind" approach (the mismatch is unknown prior to reconstruction).
<p align="center"><img src="https://github.com/mehrhardt/blind_remote_sensing/blob/master/fig_2.png" width="60%" border="0"/></p>

## Getting started
There are a number of examples which also reproduce the results as presented in the paper. To execute them all, run [matlab/example.m](matlab/example.m). It will run all examples in [matlab/src/scripts](matlab/src/scripts). These include

* [RS_algorithm_comparison_trees1_shift_5px_disk.m](matlab/src/scripts/RS_algorithm_comparison_trees1_shift_5px_disk.m)
* [RS_algorithm_comparison_trees2_ch108_NW.m](matlab/src/scripts/RS_algorithm_comparison_trees2_ch108_NW.m)
* [RS_example_blind_v_nonblind.m](matlab/src/scripts/RS_example_blind_v_nonblind.m)
* [RS_example_groundtruth_disk.m](matlab/src/scripts/RS_example_groundtruth_disk.m)
* [RS_example_groundtruth_gaussian.m](matlab/src/scripts/RS_example_groundtruth_gaussian.m)
* [RS_example_trees1_NE_spectral_comparison.m](matlab/src/scripts/RS_example_trees1_NE_spectral_comparison.m)
* [RS_example_trees2_ch108_NW.m](matlab/src/scripts/RS_example_trees2_ch108_NW.m)
* [RS_example_TV_v_dTV.m](matlab/src/scripts/RS_example_TV_v_dTV.m)
* [RS_example_urban_ch1_city.m](matlab/src/scripts/RS_example_urban_ch1_city.m)
* [RS_example_urban_park_spectral_comparison.m](matlab/src/scripts/RS_example_urban_park_spectral_comparison.m)
* [RS_example_comparison_gamma.m](matlab/src/scripts/RS_example_comparison_gamma.m)
* [RS_print_algorithm_comparison.m](matlab/src/scripts/RS_print_algorithm_comparison.m)

## Further Improvements
As suggested in [2], the code can be made more robust to large deformations in the side information by a different initialisation of the image to be reconstructed. See [2] for more details.

## References
[1] Bungert, L., Coomes, D. A., Ehrhardt, M. J., Rasch, J., Reisenhofer, R., & Schönlieb, C.-B. (2018). Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation. Inverse Problems, 34(4), 044003. https://doi.org/10.1088/1361-6420/aaaf63 http://arxiv.org/abs/1710.05705

[2] Bungert, L., Ehrhardt, M. J., & Reisenhofer, R. (2018). Robust Blind Image Fusion for Misaligned Hyperspectral Imaging Data. In Proceedings in Applied Mathematics & Mechanics (Vol. 18, p. e201800033). https://doi.org/10.1002/pamm.201800033
