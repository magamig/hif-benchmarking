function [Yhim] = downsamp_HS(Yhim_up, downsamp_factor, shift)
%downsamp_HS - the equivalent of applying matrix M

[nl, nc, L] = size(Yhim_up);
Yhim = zeros(ceil(nl/downsamp_factor), ceil(nc/downsamp_factor), L);
for i=1:L
    Yhim(:,:,i) = downsample(downsample(Yhim_up(:,:,i), downsamp_factor, shift)', downsamp_factor, shift)';
end