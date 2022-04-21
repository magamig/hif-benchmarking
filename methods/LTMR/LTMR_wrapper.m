function Out = LTMR_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls the LTMR method [1].
%
% USAGE
%       Out = LTTR_wrapper(HSI,MSI)
%
% INPUT
%       HSI   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MSI   : MS image (rows1,cols1,bands1)
%
% OUTPUT
%       Out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] Dian, R., & Li, S. (2019). Hyperspectral image super-resolution
%           via subspace-based low tensor multi-rank regularization. IEEE 
%           Transactions on Image Processing, 28(10), 5135-5146.
%--------------------------------------------------------------
% Set the default values of the parametes
%--------------------------------------------------------------

Y_h = HSI;
Y_h_bar = hyperConvert2D(Y_h);
[m,n,L] = size(Y_h);

Y = MSI;
Y_bar = hyperConvert2D(Y);
[M,N,l] = size(Y);

sz = [M N];
sf = M/n;
psf = fspecial('gaussian',7,2);
fft_B = psf2otf(psf,sz);

para.K=200;
para.eta=1e-3;
para.patchsize=7;
para.p=10;

[R,~] = estR(Y_h,Y);
for b = 1:l
    msi = reshape(Y(:,:,b),M,N);
    msi = msi - R(b,end);
    msi(msi<0) = 0;
    Y(:,:,b) = msi;
end
R = R(:,1:end-1);  
F = R;

Out = TSVD_Subpace_FUS(HSI,MSI,F,fft_B,sf,nan,para);

