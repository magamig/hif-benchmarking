function Out = FUSE_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls FUSE [1], to be used 
% for the comparisons in [2].
%
% USAGE
%       Out = FUSE_wrapper(HSI,MSI,ratio,scaling)
%
% INPUT
%       HSI    : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MSI    : MS image (rows1,cols1,bands1)
%       ratio  : GSD ratio
%       scaling: scaling factor
%
% OUTPUT
%       Out    : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] Q. Wei, N. Dobigeon, and J.-Y. Tourneret, "Fast fusion of 
%           multi-band images based on solving a Sylvester equation," IEEE 
%           Trans. Image Process., vol. 24, no. 11, pp. 4109�4121, Nov. 
%           2015.
%       [2] N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and 
%           multispectral data fusion: a comparative review of the recent 
%           literature," IEEE Geoscience and Remote Sensing Magazine, vol. 
%           5, no. 2, pp. 29-56, June 2017.
%--------------------------------------------------------------------------

ratio = size(MSI,1)/size(HSI,1);
scaling = 10000;

[rows1,cols1,bands1] = size(MSI);

[R,~] = estR(HSI,MSI);
for b = 1:bands1
    msi = reshape(MSI(:,:,b),rows1,cols1);
    msi = msi - R(b,end);
    msi(msi<0) = 0;
    MSI(:,:,b) = msi;
end
R = R(:,1:end-1);
start_pos = [round(ratio/2) round(ratio/2)];
size_kernel=[round(ratio/2)*2+1 round(ratio/2)*2+1];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
KerBlu = fspecial('gaussian',[size_kernel(1) size_kernel(2)],sig);
[Out]= BayesianFusion(HSI*scaling,MSI*scaling,R,KerBlu,ratio,'Gaussian',start_pos);
if mod(ratio,2) == 0
    start_pos = [round(ratio/2)+1 round(ratio/2)+1];
    [Out2]= BayesianFusion(HSI*scaling,MSI*scaling,R,KerBlu,ratio,'Gaussian',start_pos);
    Out = (Out+Out2)/2;
end
Out = Out/scaling;