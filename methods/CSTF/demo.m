% CSTF for Hyperspectral image and multispectral image fusion, Version 2.0
% Copyright(c) 2018 Renwei Dian
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
% resolution multispectral image.
% 
% if you use this code, Please cite the following paper:
%
% S. Li, R. Dian, L. Fang, and J. Bioucas-Dias, Fusing Hyperspectral and Multispectral Images via
% Coupled Sparse Tensor Factorization, IEEE Trans. Image Process., Vol. 27, No.8, 2018

%--------------------------------------------------------------------------









clear
clc

addpath('functions', 'ImprovedDL', 'tensor_toolbox_2.6','sisal')
S=imread('original_rosis.tif');
F=load('R.mat');


S=double(S);
S=S(1:256,1:256,11:end);
S=S/max(S(:));

F=F.R;
%  for band = 1:size(F,1)
%         div = sum(F(band,:));
%         for i = 1:size(F,2)
%             F(band,i) = F(band,i)/div;
%         end
%  end

 T=F(:,1:end-10);
    
[M,N,L] = size(S);

%%  simulate LR-HSI
S_bar = hyperConvert2D(S);
downsampling_scale=8;

s0=downsampling_scale/2;
BW=ones(8,1)/8;
 BW1=psf2otf(BW,[M 1]);
 S_w=ifft(fft(S).*repmat(BW1,1,N,L)); %blur with the width  mode
 
 
BH=ones(8,1)/8;
 BH1=psf2otf(BH,[N 1]);
aa=fft(permute(S_w,[2 1 3]));
  S_h=(aa.*repmat(BH1,1,M,L));
 S_h= permute(ifft(S_h),[2 1 3]);  %blur with the height mode
 
  Y_h=S_h(s0:downsampling_scale:end,s0:downsampling_scale:end,:);% uniform downsamping
  Y_h_bar=hyperConvert2D(Y_h);

  SNRh=35;
sigmam = sqrt(sum(Y_h_bar(:).^2)/(10^(SNRh/10))/numel(Y_h_bar));
rng(10,'twister')
   Y_h_bar = Y_h_bar+ sigmam*randn(size(Y_h_bar));
HSI=hyperConvert3D(Y_h_bar,M/downsampling_scale, N/downsampling_scale );



  %%  simulate HR-MSI
 rng(10,'twister')
Y = T*S_bar;
SNRm=40;
sigmam = sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y = Y+ sigmam*randn(size(Y));
MSI=hyperConvert3D(Y,M,N);




 par.W=240; par.H=240;  par.S=9; par.lambda=1e-5;

t=clock;
 Z4 = CSTF_FUS(HSI,MSI,T,BW1,BH1,downsampling_scale,par,s0,S);
 t4=etime(clock,t);
 [psnr4,rmse4, ergas4, sam4, uiqi4,ssim4,DD4,CC4] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z4)), 0, 1.0/downsampling_scale);







