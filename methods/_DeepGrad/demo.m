% Script for obtaining HR-HSI estimation.
%
% Reference: 
% Hyperspectral image super-resolution with deep priors and degradation model inversion
% Xiuheng Wang, Jie Chen, C¨¦dric Richard
%
% 2021/10
% Implemented by
% Xiuheng Wang
% xiuheng.wang@oca.eu

clear; clc; 
close all;
addpath('functions');

sf = 32;
kernel_type     =    {'Uniform_blur'};
Test_file       =    'jelly_beans_ms';

folder='CAVE'; % Ground truths
folder_hat = 'UAL'; % Results from base methods
folderresults = strcat('Results_', folder_hat); % Post-processed results 
Iter = 20;

if ~exist(folderresults,'file')
    mkdir(folderresults);
end  

% Hyper-parameters setups
rho = 1e-3;
mu = 5e-2 / rho; % mu' = mu / rho
nu = 5e-3 / rho;  % nu' = nu / rho

%% Load data and generate LR-HSI and HR-RGB
im_structure =load(fullfile(folder, Test_file), 'truth');
S = double(im_structure.truth) / 255.0; 
im_structure =load(fullfile(folder_hat, Test_file), 'RE');
X_hat = double(im_structure.RE)/ 255.0; 

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

%% HSI super-resolution accounting for spectral-spatial gradient deviation
W = X_hat;

[psnr, rmse, ergas, sam,ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_hat)), 0, 1.0/sf);
fprintf('Before enhancing: RMSE = %3.3f, PSNR = %2.3f, ERGAS = %2.3f, SAM = %2.2f, SSIM = %2.4f \n\n', rmse, psnr, ergas, sam, ssim);

for j = 1:Iter
    % Optimization w.r.t. X
    HR_HSI3=hyperConvert2D(W);
    H1=(R'*R) + rho*eye(size(R,2));
    HHH1=par.HT(HSI3);
    H3=(R'*MSI3)+rho*HR_HSI3+HHH1;
    X_fin=Sylvester(H1,par.fft_B,sf,nr/sf,nc/sf,H3);
    X_fin3=hyperConvert3D(X_fin, nr, nc );
    [psnr, rmse, ergas, sam, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_fin3)), 0, 1.0/sf);
    fprintf('Iter = %2.0f:\nRMSE = %3.3f, PSNR = %2.3f, ERGAS = %2.3f, SAM = %2.2f, SSIM = %2.4f \n\n', j, rmse, psnr, ergas, sam, ssim);
    
    % Optimization w.r.t. v
    W = deconvs_auto(X_fin3, X_hat, mu, nu); 
end

[psnr, rmse, ergas, sam, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(X_fin3)), 0, 1.0/sf);
fprintf('After enhancing: RMSE = %3.3f, PSNR = %2.3f, ERGAS = %2.3f, SAM = %2.2f, SSIM = %2.4f \n\n', rmse, psnr, ergas, sam, ssim);
fprintf('========================================================================================\n\n');

%% Store data
sr = X_fin3; 
save(fullfile(folderresults, strcat(Test_file, '.mat')), 'sr');
