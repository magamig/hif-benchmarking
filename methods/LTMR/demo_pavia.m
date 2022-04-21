
clear
clc

addpath(genpath('LTMR'))

 
S=imread('./data/original_rosis.tif');

S=S(1:256,1:256,11:end);
S=double(S);
S=S/max(S(:));
[M N L]=size(S);

sf =4;
sz=[M N];
s0=1;
 psf        =    fspecial('gaussian',7,2);
  par.fft_B      =    psf2otf(psf,sz);
  par.fft_BT     =    conj(par.fft_B);
par.H          =    @(z)H_z(z, par.fft_B, sf, sz,s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz,s0);

% R=load('C:\Users\Utilizador\Desktop\drw\learningcode\HSI-superresolution\TGRS-2015\HySure-master1\HySure-master\data\ikonos_spec_resp.mat');
% R=R.ikonos_sp;
% [~, valid_ik_bands] = intersect(R(:,1), 430:860);
% no_wa = length(valid_ik_bands);
% xx  = linspace(1, no_wa, L);
% x = 1:no_wa;
% F = zeros(5, L);
% for i = 1:5 % 1 - pan; 2 - blue; 3 - green; 4 - red; 5 - NIR
%     F(i,:) = spline(x, R(valid_ik_bands,i+1), xx);
% end
% F = F(2:4,:);
load('./data/R.mat');


F=R;
F=F(:,1:end-10);
for band = 1:size(F,1)
        div = sum(F(band,:));
        for i = 1:size(F,2)
            F(band,i) = F(band,i)/div;
        end
end
S_bar = hyperConvert2D(S);
hyper= par.H(S_bar);
MSI = hyperConvert3D((F*S_bar), M, N);
  HSI =hyperConvert3D(hyper,M/sf, N/sf );

 
  
%% Hysure
%  basis_type = 'VCA';
% lambda_phi = 5e-5;
% lambda_m = 1;
% p=10;
% B=ifft2( par.fft_B );
% t0=clock;
%  Z1=data_fusion( HSI, MSI, sf, F, B, p, basis_type, lambda_phi, lambda_m,s0 );
% t1=etime(clock,t0)
%  [psnr1,rmse1, ergas1, sam1, uiqi1,ssim1,DD1,CC1] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z1)), 0, 1.0/sf);

 
 
%% CSU

% par.w=size(S,1);
% par.h=size(S,2);
% p=10;
% t0=clock;
% [E,A] = SupResPALM(HSI, MSI, S_bar, F,p,par);
%  Z1 = hyperConvert3d(E*A);
% t1=etime(clock,t0)
%  [psnr1,rmse1, ergas1, sam1, uiqi1,ssim1,DD1,CC1] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z1)), 0, 1.0/sf);

%% NSSR
% par.P=F;
% par.w=size(S,1);
% par.h=size(S,2);
% par.eta2       =  1e-4;    % 0.03
%     par.eta1       =   1e-4;
%     par.mu         =  2e-4;   % 0.004
%     par.ro         =   1.1; 
%     par.Iter       =   26;
% par.K          =    80;
% par.lambda     =    0.001;
% par.s0=s0;
% t0=clock;
% Z2     =    NSSR_HSI_SR1( HSI,MSI,S_bar, sf,par,sz,s0 );
% Z2=hyperConvert3D(Z2,sz(1),sz(2));
% t2=etime(clock,t0)
%  [psnr2,rmse2, ergas2, sam2, uiqi2,ssim2,DD2,CC2] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z2)), 0, 1.0/sf);

%% CSTF
% par1.W=260; par1.H=260;  par1.S=15; par1.lambda=1e-5;
% BW       =    fspecial('gaussian',[7 1],2);
%  BW1=psf2otf(BW,[M 1]);
%  BH       =    fspecial('gaussian',[7 1],2);
%  BH1=psf2otf(BH,[N 1]);
% t0=clock;
%  Z3 = CSTF_FUS(HSI,MSI,F,BW1,BH1,sf,par1,s0,S);
%   t3=etime(clock,t0)
%  [psnr3,rmse3, ergas3, sam3, uiqi3,ssim3,DD3,CC3] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z3)), 0, 1.0/sf);

%% LTMR

para.K=200;
 para.eta=1e-3;
 para.patchsize=7;
para.p=10;
t0=clock;
 Z4 = TSVD_Subpace_FUS(HSI,MSI,F, par.fft_B,sf,S,para);
  t4=etime(clock,t0)
 [psnr4,rmse4, ergas4, sam4, uiqi4,ssim4,DD4,CC4] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z4)), 0, 1.0/sf);
% 
% QM.S=S;
% QM.Z1=Z1;
% QM.Z2=Z2;
% QM.Z3=Z3;
% QM.Z4=Z4;
% 
% 
% 
% 
% QM.psnr=[mean(psnr1), mean(psnr2),mean(psnr3),mean(psnr4)];
% QM.rmse=[mean(rmse1), mean(rmse2),mean(rmse3),mean(rmse4)];
% QM.t=[mean(t1), mean(t2),mean(t3),mean(t4)];
% QM.sam=[mean(sam1), mean(sam2),mean(sam3),mean(sam4)];
% QM.ergas=[mean(ergas1), mean(ergas2),mean(ergas3),mean(ergas4)];
% QM.uiqi=[mean(uiqi1), mean(uiqi2),mean(uiqi3),mean(uiqi4)];
% % 

