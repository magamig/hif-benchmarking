function [out] = var_dim(X,P)
out=reshape(reshape(X,[size(X,1)*size(X,2) size(X,3)])*P',[size(X,1) size(X,2) size(P,1)]);
