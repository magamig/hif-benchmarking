function [ssim1] = GetSSIMofHSI( oI,I,p,q )
%   Calculate the average SSIM score of the two hyperspectral images
% oI: the original image of size [p, q, band]
% I: the recovery image of size [p, q, band]
% p: row
% q: col

[m,n] = size(oI);
[m1,n1] = size(I);
if m1 ~= m || n1~=n
    error('Two image with different size!')
    ssim1 = -Inf;
    return;
end
window = fspecial('gaussian', 11, 1.5);	%
K(1) = 0.01;					% default settings
K(2) = 0.03;					%
L = 1; 
score = 0;
for i = 1 : m
    img1 = reshape(oI(i,:),p,q);
    img2 = reshape(I(i,:),p,q);
    score = score + ssim(img1 .* 255, img2 .* 255);
end
ssim1 = score * 1.0 / m;

end

