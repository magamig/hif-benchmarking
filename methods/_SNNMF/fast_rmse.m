function [ val ] = fast_rmse(Z,rZ)
%FAST_RMSE Summary of this function goes here
%   Detailed explanation goes here
val = 255*sqrt((norm(Z-rZ,'fro')^2)/(size(Z,1)*size(Z,2)));
end

