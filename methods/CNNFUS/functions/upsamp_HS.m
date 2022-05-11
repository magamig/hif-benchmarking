function [Yhim_up] = upsamp_HS(Yhim, downsamp_factor, nl, nc, shift)
%upsamp_HS - convert image Ymim to an image matrix with the same size as 
% Ymim. The result is a matrix filled with zeros.

[nlh, nch, L] = size(Yhim);
aux = zeros(nlh*downsamp_factor, nch*downsamp_factor, L);
for i=1:L
    aux(:,:,i) = upsample(upsample(Yhim(:,:,i), downsamp_factor, shift)', downsamp_factor, shift)';
end
Yhim_up = aux(1:nl, 1:nc, :);
