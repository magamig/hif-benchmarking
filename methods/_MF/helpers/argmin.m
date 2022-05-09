function [i,j] = argmin(M)

[Y, I] = min(M,[],1);
[dc, j] = min(Y);
i = I(j);