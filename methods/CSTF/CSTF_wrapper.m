function Out = CSTF_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls the CSTF method [1].
%
% USAGE
%       Out = CSTF_wrapper(HSI,MSI)
%
% INPUT
%       HSI   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MSI   : MS image (rows1,cols1,bands1)
%
% OUTPUT
%       Out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] Li, S., Dian, R., Fang, L., & Bioucas-Dias, J. M. (2018). 
%           Fusing hyperspectral and multispectral images via coupled 
%           sparse tensor factorization. IEEE Transactions on Image 
%           Processing, 27(8), 4118-4130.
%--------------------------------------------------------------
% Set the default values of the parametes
%--------------------------------------------------------------


[m,n,L] = size(HSI);
[M,N,l] = size(MSI);

sz = [M N];
sf = M/m;
s0 = sf/2;
BW = fspecial('gaussian', [7 1], 2);
BW1 = psf2otf(BW,[M 1]);
BH = fspecial('gaussian', [7 1], 2);
BH1 = psf2otf(BH,[N 1]);

par1.W=M;
par1.H=N;
par1.S=L;
par1.lambda=1e-5;

[R,~] = estR(HSI,MSI);
for b = 1:l
    msi2 = reshape(MSI(:,:,b),M,N);
    msi2 = msi2 - R(b,end);
    msi2(msi2<0) = 0;
    MSI(:,:,b) = msi2;
end
R = R(:,1:end-1);  
F = R;

Out = CSTF_FUS(HSI,MSI,F,BW1,BH1,sf,par1,s0,nan);
