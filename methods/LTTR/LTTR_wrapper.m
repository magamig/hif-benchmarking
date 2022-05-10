function Out = LTTR_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls the LTTR method [1].
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
%       [1] R. Dian, S. Li, and  L. Fang, Learning a Low Tensor-Train Rank
%           Representation for Hyperspectral Image Super-Resolution, 
%           IEEE TNNLS, 2019
%--------------------------------------------------------------
% Set the default values of the parametes
%--------------------------------------------------------------

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

Y_h = HSI;
Y_h_bar = hyperConvert2D(Y_h);
[m,n,L] = size(Y_h);

Y = MSI;
Y_bar = hyperConvert2D(Y);
[M,N,l] = size(Y);

sz = [M N];
sf = M/m;
psf = fspecial('gaussian',7,2);
fft_B = psf2otf(psf,sz);
K=3;
eta=1e-2;

[R,~] = estR(Y_h,Y);
for b = 1:l
    msi = reshape(Y(:,:,b),M,N);
    msi = msi - R(b,end);
    msi(msi<0) = 0;
    Y(:,:,b) = msi;
end
R = R(:,1:end-1);  
F = R;

Out = LTTR_FUS(Y_h,Y,F,K,eta,fft_B,sf,nan);
