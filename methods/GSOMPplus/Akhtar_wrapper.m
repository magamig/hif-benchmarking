function Out = Akhtar_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls Akhtar's method [1], to be used for
% the comparisons in [2].
%
% USAGE
%       Out = Akhtar_wrapper(HSI,MSI)
%
% INPUT
%       HSI   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MSI   : MS image (rows1,cols1,bands1)
%
% OUTPUT
%       Out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] Akhtar, Naveed, Faisal Shafait, and Ajmal Mian. "Sparse Spatio-
%           spectral Representation for Hyperspectral Image 
%           Super-Resolution." Computer Vision�ECCV 2014. Springer 
%           International Publishing, 2014. 63-78.
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
%---------------------------------------------
% Setting the parameters to the default values
%-----------------------------------------------
param.spams = 1;        % Set = 1 if SPAMS***(see below)is installed, 0 otherwise
param.L = 20;           % Atoms selected in each iteration of G-SOMP+
param.gamma = 0.99;     % Residual decay parameter
param.k = bands2;          % Number of dictionary atoms
param.eta = 10e-3;      % Modeling error
patchsize = rows2;

[R,~] = estR(HSI,MSI);
for b = 1:bands1
    msi = reshape(MSI(:,:,b),rows1,cols1);
    msi = msi - R(b,end);
    msi(msi<0) = 0;
    MSI(:,:,b) = msi;
end
R = R(:,1:end-1);
[Out] = superResolution2(param,HSI,MSI,R,patchsize);
if maxval > 1
    Out = Out * maxval;
end