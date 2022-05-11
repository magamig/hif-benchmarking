
clear
clc
addpath(genpath('.'))

S=imread('data/original_rosis.tif');
F=load('data/R.mat');
S=double(S);
S=S(:,:,11:end);
S=S/max(S(:));

F=F.R;
 F=F(:,1:end-10);
 for band = 1:size(F,1)
        div = sum(F(band,:));
        for i = 1:size(F,2)
            F(band,i) = F(band,i)/div;
        end
 end


    
[M,N,L] = size(S);

%%  simulate LR-HSI
S_bar = hyperConvert2D(S);
downsampling_scale=5;
psf        =    fspecial('gaussian',7,2);
par.fft_B      =    psf2otf(psf,[M N]);
par.fft_BT     =    conj(par.fft_B);
s0=1;
par.H          =    @(z)H_z(z, par.fft_B, downsampling_scale, [M N],s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, downsampling_scale,  [M N],s0);
Y_h_bar=par.H(S_bar);

  
SNRh=30;
sigma = sqrt(sum(Y_h_bar(:).^2)/(10^(SNRh/10))/numel(Y_h_bar));
rng(10,'twister')
   Y_h_bar = Y_h_bar+ sigma*randn(size(Y_h_bar));
HSI=hyperConvert3D(Y_h_bar,M/downsampling_scale, N/downsampling_scale );



  %%  simulate HR-MSI
rng(10,'twister')
Y = F*S_bar;
SNRm=35;
sigmam = sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y = Y+ sigmam*randn(size(Y));
MSI=hyperConvert3D(Y,M,N);



%% CNN_FUS
para.gama=1.1;
para.p=10;
para.sig=10e-4;
t0=clock;
 [Z6]= CNN_Subpace_FUS( HSI, MSI,F,par.fft_B,downsampling_scale,S,para,1);
 t6=etime(clock,t0)
 [psnr6,rmse6, ergas6, sam6, uiqi6,ssim6,DD6,CC6] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z6)), 0, 1.0/downsampling_scale);


%t=[t1 t2 t3 t4 t5 t6];
save('pavia.mat','Z6')




