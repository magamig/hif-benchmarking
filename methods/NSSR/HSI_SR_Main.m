% =========================================================================
% NSSR for Hyperspectral image super-resolution, Version 1.0
% Copyright(c) 2016 Weisheng Dong
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for Hyperspectral image super-
% resolution from a pair of low-resolution hyperspectral image and a high-
% resolution RGB image.
% 
% Please cite the following paper if you use this code:
%
% Weisheng Dong, Fazuo Fu, et. al.,"Hyperspectral image super-resolution via 
% non-negative structured sparse representation", IEEE Trans. On Image Processing, 
% vol. 25, no. 5, pp. 2337-2352, May 2015.
% 
%--------------------------------------------------------------------------

clc;
clear;
Dir             =    'Data/CAVE';
Result_dir      =    'Results/CAVE_Results/';
Test_file       =    {'chart_and_stuffed_toy_ms2', 'oil_painting_ms', 'cloth_ms', 'fake_and_real_peppers_ms'};
kernel_type     =    {'Uniform_blur', 'Gaussian_blur'};
pre             =   'NSSR_';
sf              =    8;
Out_dir         =    fullfile(Result_dir, sprintf('sf_%d',sf));


[Z_res, RMSE, PSNR, sz]     =    NSSR_HSI_SR( Dir, Test_file{1}, sf, kernel_type{2} );
Dir_out                     =    fullfile( Out_dir, Test_file{1} );
if exist(Dir_out,'dir')==0
    mkdir( Dir_out );
end

Save_HSI( Z_res, sz, Dir_out, strcat(pre,Test_file{1}) );

disp( sprintf('Scaling factor = %d,  %s,  RMSE = %3.3f, PSNR = %2.3f \n', sf, Test_file{1}, RMSE, PSNR));

