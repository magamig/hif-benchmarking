% LTTR for Hyperspectral image and multispectral image fusion, Version 2.0
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
%  R. Dian, S. Li, and  L. Fang, Learning a Low Tensor-Train Rank
% Representation for Hyperspectral Image Super-Resolution, IEEE TNNLS, 2019

clear
clc

addpath(genpath('LTTR_file'))

F=create_F();
sf = 8;


sz=[512 512];
s0=1;
 psf        =    fspecial('gaussian',7,2);
  par.fft_B      =    psf2otf(psf,sz);
  par.fft_BT     =    conj(par.fft_B);
par.H          =    @(z)H_z(z, par.fft_B, sf, sz,s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz,s0);
par.P=create_F();
F=F(:,3:31);
 for band = 1:size(F,1)
        div = sum(F(band,:));
        for i = 1:size(F,2)
            F(band,i) = F(band,i)/div;
        end
    end




%% CSU
% for yy=1:32 
% im_structure =load(fullfile(pathstr, 'shuju1', imglist(yy).name));
% S = im_structure.b;           
% [M,N,L] = size(S);
% S_bar = hyperConvert2D(S);
% hyper= par.H(S_bar);
% multi=F*S_bar;
% par.w=size(S,1);
% par.h=size(S,2);
% p=10;
% t0=clock;
% [E,A] = SupResPALM(hyper, multi, S_bar, F,p,par);
%  Z = hyperConvert3d(E*A);
% t1(yy)=etime(clock,t0)
%  [psnr1(yy),rmse1(yy), ergas1(yy), sam1(yy), uiqi1(yy),ssim1(yy),DD1(yy),CC1(yy)] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1.0/sf);
%  end

%% NSSR
% for yy=1:32 
% im_structure =load(fullfile(pathstr, 'shuju1', imglist(yy).name));
% S = im_structure.b;           
% [M,N,L] = size(S);
% S_bar = hyperConvert2D(S);
% hyper= par.H(S_bar);
% Y_h = hyperConvert3D(hyper, M/sf, N/sf);
% Y = hyperConvert3D((F*S_bar), M, N);
% par.P=F;
% par.w=size(S,1);
% par.h=size(S,2);
% par.eta2       =  1e-4;    % 0.03
%     par.eta1       =   1e-2;
%     par.mu         =  2e-4;   % 0.004
%     par.ro         =   1.1; 
%     par.Iter       =   26;
% par.K          =    80;
% par.lambda     =    0.001;
% par.s0=s0;
% t0=clock;
% Z2     =    NSSR_HSI_SR1( Y_h,Y,S_bar, sf,par,sz,s0 );
% Z2=hyperConvert3D(Z2,sz(1),sz(2));
% t2(yy)=etime(clock,t0)
%  [psnr2(yy),rmse2(yy), ergas2(yy), sam2(yy), uiqi2(yy),ssim2(yy),DD2(yy),CC2(yy)] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z2)), 0, 1.0/sf);
%  end






%% NLSTF
%  for yy=1:32
%     yy
% im_structure =load(fullfile(pathstr, 'shuju1', imglist(yy).name));
% S = im_structure.b;           
% [M,N,L] = size(S);
% S_bar = hyperConvert2D(S);
% hyper= par.H(S_bar);
% Y_h = hyperConvert3D(hyper, M/sf, N/sf);
% Y = hyperConvert3D((F*S_bar), M, N);
% K=160;
% C=0.012;
% t0=clock;
%  Z = LTTR_FUS(Y_h,Y,F,K,C, par.fft_B,sf,S);
%   t4(yy)=etime(clock,t0)
%  [psnr4(yy),rmse4(yy), ergas4(yy), sam4(yy), uiqi4(yy),ssim4(yy),DD4(yy),CC4(yy)] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1.0/sf);
%  end




im_structure =load('.\data\face_ms.mat');
S = im_structure.b; 
S=S(:,:,3:31);
[M,N,L] = size(S);
S_bar = hyperConvert2D(S);
hyper= par.H(S_bar);
Y_h = hyperConvert3D(hyper, M/sf, N/sf);
Y = hyperConvert3D((F*S_bar), M, N);
para.K=160;
para.eta=1e-2;
t0=clock;
 Z = LTTR_FUS(Y_h,Y,F,para.K,para.eta, par.fft_B,sf,S);
  t4=etime(clock,t0)
 [psnr4,rmse4, ergas4, sam4, uiqi4,ssim4,DD4,CC4] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1.0/sf);








