function [ rmse_total ] = getrmse( x,y )
% x=double(im2uint8(x));
% y=double(im2uint8(y));
sz_x = size(x);
n_bands = sz_x(3);
n_samples = sz_x(1)*sz_x(2);
aux = sum(sum((x - y).^2, 1), 2)/n_samples;
rmse_total = sqrt(sum(aux, 3)/n_bands);
end

