function [Y,SNR_dB] = denoising(X,mask)
%--------------------------------------------------------------
% Denoising of HSI
%
% USAGE
%     [Y,SNR_dB] = denoising(X)
% INPUT
%     X      : hyperspectral data (rows,cols,bands)
% OUTPUT
%     Y      : denoised hyperspectral data (rows,cols,bands)
%     SNR_dB : estimated SNR in dB
%
% REFERENCE
%     R. Roger, "Principal components transform with simple automatic noise
%     adjustment," International Journal of Remote Sensing, vol. 17, pp.
%     2719-2727, 1996
%
% Author: Naoto YOKOYA
% Email : yokoya@sal.rcast.u-tokyo.ac.jp
%--------------------------------------------------------------
[rows,cols,bands] = size(X);
SNR_dB = zeros(1,bands);
Y = X;
if nargin == 1
    for i = 1:bands
        %disp([num2str(i) 'th band']);
        x = reshape(X(:,:,i),[],1);
        if i == 1
            A = reshape(X(:,:,i+1:end),[],bands-1);
        elseif i == bands
            A = reshape(X(:,:,1:end-1),[],bands-1);
        else
            A = reshape(X(:,:,[1:i-1 i+1:end]),[],bands-1);
        end
        invAtA = pinv(A'*A);
        Y(:,:,i) = reshape((A*invAtA)*(A'*x),rows,cols);
        SNR_dB(i) = 20*log10(mean(x)/mean((x-reshape(Y(:,:,i),[],1)).^2)^0.5);
    end
else
    for i = 1:bands 
        %disp([num2str(i) 'th band']);
        x0 = reshape(X(:,:,i),[],1);
        if i == 1
            A0 = reshape(X(:,:,i+1:end),[],bands-1);
        elseif i == bands
            A0 = reshape(X(:,:,1:end-1),[],bands-1);
        else
            A0 = reshape(X(:,:,[1:i-1 i+1:end]),[],bands-1);
        end
        x = x0(mask,1);
        A = A0(mask,:);
        invAtA = pinv(A'*A);
        
        c = (invAtA)*(A'*x);
        Y(:,:,i) = reshape(A0*c,rows,cols);
        SNR_dB(i) = 20*log10(mean(x0)/mean((x0-reshape(Y(:,:,i),[],1)).^2)^0.5);
    end
end