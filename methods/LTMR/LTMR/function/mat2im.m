function [A] = mat2im(X, nl)
%mat2im - converts a matrix to a 3D image

[p, n] = size(X);
nc = n/nl;
A = reshape(X', nl, nc, p);