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

function    [HSI_res, RMSE, PSNR, sz]    =    NSSR_HSI_SR( Dir, image, sf, kernel_type )
rand('seed',0);
time0           =    clock;
[Z_ori,sz]      =    load_HSI( Dir, image, sf );
par             =    NSSR_Parameters_setting( sf, kernel_type, sz );
X               =    par.H(Z_ori);
par.P           =    create_P();
Y               =    par.P*Z_ori;

D               =    Nonnegative_DL( X, par );   
D0              =    par.P*D;
N               =    Comp_NLM_Matrix( Y, sz );   

HSI_res         =    Nonnegative_SSR( D, D0, X, Y, N, par, Z_ori, sf, sz );

MSE             =    mean( mean( (Z_ori-HSI_res).^2 ) );
PSNR            =    10*log10(1/MSE);       
RMSE            =    sqrt(MSE)*255;
disp( sprintf('The final RMSE = %3.3f, PSNR = %3.2f', RMSE, PSNR) );   
disp(sprintf('Total elapsed time = %f min\n', (etime(clock,time0)/60) ));


