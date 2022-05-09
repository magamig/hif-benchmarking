% DHSIS for Hyperspectral image sharpening, Version 1.0
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
% Please cite the following paper if you use this code:
%
% R. Dian, S. Li, A. Guo, and L. Fang, “Deep hyperspectral image
% sharpening,” IEEE Trans. Neural Netw. Learn. Syst., to be published,
% 2018, DOI:10.1109/TNNLS.2018.2798162.
% 
%--------------------------------------------------------------------------




clear
clc
addpath('function')

sf              =    8;

t1=clock;
  mu=5e-5;

       

aaaa1=strcat('.\data\ground_truth.mat');
im_structure1 =load(aaaa1);
S=im_structure1.b;
 [nr,nc,L] = size(S);
S_bar=Unfold(S,size(S),3);
 R=create_F();



  %% genertate LR-HSI
kernel_type     =    {'Gaussian_blur'};
sz=[nr nc];
s0=1;
par             =    Parameters_setting( sf, kernel_type, sz );
 HSI3                =    par.H(S_bar);
 SNRm=30;
 sigmam = sqrt(sum(HSI3(:).^2)/(10^(SNRm/10))/numel(HSI3));
HSI3 = HSI3+ 0*randn(size(HSI3));
 HSI=hyperConvert3D(HSI3,nr/sf, nc/sf );

  %% genertate HR-MSI
SNRm=30;
MSI = hyperConvert3D((R*S_bar), nr, nc);
 sigmam = sqrt(sum(MSI(:).^2)/(10^(SNRm/10))/numel(MSI));
MSI = MSI+ 0*randn(size(MSI));
MSI1=Unfold(MSI,size(MSI),1);
MSI2=Unfold(MSI,size(MSI),2);
MSI3=Unfold(MSI,size(MSI),3);


  %% Inlize the HR-HSI from the fusing framework
   H1=R'*R+mu*eye(size(R,2));
   HR_load1=imresize(HSI,sf,'bicubic');
  HR_HSI3=hyperConvert2D(HR_load1);
   HHH1=par.HT(HSI3)  ;
 H3=R'*MSI3+mu*HR_HSI3+HHH1;
X_in=Sylvester(H1,par.fft_B ,sf,nr/sf,nc/sf,H3); %% Sylvester equation (5)
X_in3=hyperConvert3D(X_in,nr,nc);




 %% get the residual from the CNN ?for the CNN, please refer to eval_deep_cnn.py ?
aaaa1=strcat('.\data\residual.mat');
X_res =load(aaaa1);
X_CNN=X_res.b+X_in3;


%% return the priors to the fusing framwwork
  H1=R'*R+mu*eye(size(R,2));
  HR_HSI3=hyperConvert2D(X_CNN);
   HHH1=par.HT(HSI3)  ;
 H3=R'*MSI3+mu*HR_HSI3+HHH1;
X_fin=Sylvester(H1,par.fft_B ,sf,nr/sf,nc/sf,H3); %% Sylvester equation (12)
X_fin3=hyperConvert3D(X_fin,nr, nc );



 [psnr1,rmse1, ergas1, sam1, uiqi1,ssim1] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_CNN)), 0, 1.0/sf);
 [psnr2,rmse2, ergas2, sam2, uiqi2,ssim2] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_fin3)), 0, 1.0/sf);


t2=etime(clock,t1);
