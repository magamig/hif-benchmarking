function [PSNR] = GetPSNR( oI,I )

% Calculate the PSNR of two HSIs which has been rescaled into [0,1]
%  oI: the original image
%   I: the recovery image

[m,n] = size(oI);
[m1,n1] = size(I);
if m1 ~= m || n1~=n
    error('Two image with different size!')
    ssim1 = -Inf;
    return;
end

MSE = sum(sum((oI - I).^2)) * 1.0 / (m * n);
PSNR = 20 * log10(2^1 - 1) - 10 * log10(MSE);

end