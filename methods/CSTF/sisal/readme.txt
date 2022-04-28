% This package contains a matlab implementation of the SISAL
% algorithm [1].
%
%
%--------------------------------------------------------------------------
%   Files included
%-------------------------------------------------------------------------
%
%  sisal.m   -> SISAL algorithm [1]
%  mvsa.m    -> MVSA algorithm [2]
%  vca.m     -> VCA algorithm [3]
%  estNoise  -> Noise estimation algorithm. See [4]
%  dataProj  -> Project data algorithm
%  hysime    -> Hysime algorithm [5]
%  USGS_1995_Library.mat -> USGS spectral library
%
%  DEMOS:
%
%  demo_sisal.m             -> basic demo
%  demo_sisal_large_n.m     -> large data set
%  demo_sisal_large_n_p.m   -> large data set and 15 endmembers
%  demo_sisal_noise_comparison.m  ->  illustration of  sisal robustness to
%                                     noise
%  demo_sisal_noise_comparison.m  ->  illustration of  sisal robustness to
%                                     outliers
%
%--------------------------------------------------------------------------
%    How to run
%-------------------------------------------------------------------------
%
%  Simply download the complete package to a directory and run the demos
%
%
% SISAL: Simplex identification via split augmented Lagrangian
%
% [1] J. Bioucas-Dias, "A variable splitting augmented Lagrangian approach
%     to linear spectral unmixing", in  First IEEE GRSS Workshop on
%     Hyperspectral Image and Signal Processing-WHISPERS'2009, Grenoble,
%     France,  2009. Available at http://arxiv.org/abs/0904.4635v
%
% MVSA: Minimum volume simple analysis
%
% [2] Jun Li and José M. Bioucas-Dias
%     "Minimum volume simplex analysis: A fast algorithm to unmix hyperspectral data"
%      in IEEE International Geoscience and Remote sensing Symposium
%      IGARSS’2008, Boston, USA,  2008.
%
% VCA: Vertex component analysis
%
% [3] J. Nascimento and J. Bioucas-Dias, "Vertex component analysis",
%     IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4,
%     pp. 898-910, 2005.
%
% [4] J. Bioucas- Dias and J. Nascimento, "Hyperspectral subspace
%     identification", IEEE Transactions on Geoscience and Remote Sensing,
%     vol. 46, no. 8, pp. 2435-2445, 2008
%
%
% NOTE:  VCA (Vertex Component Analysis) is used to initialize SISAL. However,
%        VCA is a pure-pixel based algorithm and thus it is not suited to
%        the data sets herein considered.  Nevertheless, we plot VCA results,
%        to highlight the advantage of non-pure-pixel based algorithms over the
%        the pure-pixel based ones.
%
%
% Author: Jose M. Bioucas-Dias (December, 2009)
%
