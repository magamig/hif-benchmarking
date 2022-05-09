function [A] = ConvC(X, FK, nl)
%ConvC - defines a circular convolution (the same for all bands) accepting 
% a matrix and returnig a matrix. FK is the fft2 of a one-band filter 

[p, n] = size(X);
nc = n/nl;
A = reshape(real(ifft2(fft2(reshape(X', nl, nc, p)).*repmat(FK,[1, 1, p]))), nl*nc, p)';
