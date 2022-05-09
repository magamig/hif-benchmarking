% Script for obtaining final HR-HSI estimation.
%
% Reference: 
% Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation
% Xiuheng Wang, Jie Chen, Qi Wei, C¨¦dric Richard
%
% 2019/05
% Implemented by
% Xiuheng Wang
% xiuheng.wang@mail.nwpu.edu.cn

clear; clc; 
close all;
addpath('enhancement');

sf = 8;
kernel_type     =    {'Gaussian_blur'};
% kernel_type     =    {'Uniform_blur'};

foldercnn='results';
folderresults = 'results_final';

rmse = zeros(12, 1);
psnr = zeros(12, 1);
ergas = zeros(12, 1);
sam = zeros(12, 1);
ssim = zeros(12, 1);

for i = 1:12

img=load(fullfile(foldercnn, num2str(i-1)), 'sr');
X_CNN = double(img.sr);
S=load(fullfile(foldercnn, num2str(i-1)), 'gt');
S = S.gt;
[nr,nc,L] = size(S);
S_bar=Unfold(S,size(S),3);
R=create_F();
    
sz = [nr nc];
par             =    Parameters_setting( sf, kernel_type, sz );
HSI3                =    par.H(S_bar);
HSI=hyperConvert3D(HSI3,nr/sf, nc/sf );

MSI = hyperConvert3D((R*S_bar), nr, nc);
MSI1=Unfold(MSI,size(MSI),1);
MSI2=Unfold(MSI,size(MSI),2);
MSI3=Unfold(MSI,size(MSI),3);

[X_fin, mu_opti, ita_opti]=search_2_gss(par, R, X_CNN, HSI3, MSI3, sf);

fprintf('mu_opti = %2.4e\n', mu_opti);
X_fin3=hyperConvert3D(X_fin, nr, nc );

[psnr(i), rmse(i), ergas(i), sam(i),ssim(i)] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_CNN)), 0, 1.0/sf);
fprintf('Before enhancing: RMSE = %3.3f, PSNR = %2.3f, ERGAS = %2.4f, SAM = %2.3f, SSIM = %2.5f \n\n', rmse(i), psnr(i), ergas(i), sam(i), ssim(i));

[psnr(i), rmse(i), ergas(i), sam(i),ssim(i)] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_fin3)), 0, 1.0/sf);
fprintf('After enhancing: RMSE = %3.3f, PSNR = %2.3f, ERGAS = %2.4f, SAM = %2.3f, SSIM = %2.5f \n\n', rmse(i), psnr(i), ergas(i), sam(i), ssim(i));

sr = X_fin3; 
gt = S; 
save(fullfile(folderresults, strcat(num2str(i-1), '.mat')), 'sr', 'gt');
end
fprintf('Mean: RMSE = %2.2f, PSNR = %2.2f, ERGAS = %2.3f, SAM = %2.2f\n\n',mean(rmse), mean(psnr), mean(ergas), mean(sam));
fprintf('Std: RMSE = %2.2f, PSNR = %2.2f, ERGAS = %2.3f, SAM = %2.2f\n\n',std(rmse), std(psnr), std(ergas), std(sam));