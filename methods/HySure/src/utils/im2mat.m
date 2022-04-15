function [A] = im2mat(X)
%im2mat - converts a 3D image to a matrix

[nl, nc, p] = size(X);
A = reshape(X, nl*nc, p)';