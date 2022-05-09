clear,clc,close all;
addpath(genpath('./Utilities/'));

sf = 8;     % scaling factor
kernel_type = 'Uniform_blur';       % Uniform_blur or Gaussian_blur
par = ParSet_new(sf,[512,512],kernel_type);

dataZ = load(['./data/dataZofFake_and_real_lemons.mat']);
RZ = dataZ.dataZ;
RZ = im2double(RZ);
RZ2d = loadHSI(RZ);
rzSize = size(RZ);
sz = [rzSize(1),rzSize(2)];
X = par.H(RZ2d);         % X: low resolution HSI
H = create_H(sz,sf);     % H: X = ZH
P = create_P();          % P: Y = PZ
Y = P*RZ2d;              % Y: high spatial resolution RGB image
% super-resolution
[PSNR, RMSE,SAM,SSIM,Z3d] = SRLRTS_sm(RZ2d,rzSize,sf,par,X,Y,H,P,kernel_type); % Z3d: the super-resolution result
fprintf('PSNR= %.4f, RMSE=%.4f, SAM=%.4f, SSIM=%.4f\n', PSNR, RMSE,SAM,SSIM);
