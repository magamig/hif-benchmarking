function Out = Lanaras_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls Lanaras's method [1], to be used 
% for the comparisons in [2].
%
% USAGE
%       Out = Lanaras_wrapper(HSI,MSI)
%
% INPUT
%       HSI   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MSI   : MS image (rows1,cols1,bands1)
%       REF   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% OUTPUT
%       Out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] C. Lanaras, E. Baltsavias, K. Schindler, "Hyperspectral 
%           Super-Resolution by Coupled Spectral Unmixing," In ICCV, 
%           Santiago, Chile, December 2015.
%       [2] N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and 
%           multispectral data fusion: a comparative review of the recent 
%           literature," IEEE Geoscience and Remote Sensing Magazine, vol. 
%           5, no. 2, pp. 29-56, June 2017.
%--------------------------------------------------------------------------

[rows1,cols1,bands1] = size(MSI);
[rows2,cols2,bands2] = size(HSI);

maxval = max([max(HSI(:)) max(MSI(:))]);
if maxval > 1
    HSI = HSI/maxval;
    MSI = MSI/maxval;
end
[R,~] = estR(HSI,MSI);
for b = 1:bands1
    msi = reshape(MSI(:,:,b),rows1,cols1);
    msi = msi - R(b,end);
    msi(msi<0) = 0;
    MSI(:,:,b) = msi;
end
R = R(:,1:end-1);  
[E,A] = SupResPALM(reshape(HSI,rows2*cols2,[])', reshape(MSI,rows1*cols1,[])', R, 30, rows1);
Out = permute(reshape(E*A,[],rows1,cols1),[2 3 1]);
if maxval > 1
    Out = Out * maxval;
end