function Out = NSSR_wrapper(HSI,MSI)
%--------------------------------------------------------------------------
% This is a wrapper function that calls the NSSR method [1].
%
% USAGE
%       Out = NSSR_wrapper(HSI,MSI)
%
% INPUT
%       HSI   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MSI   : MS image (rows1,cols1,bands1)
%
% OUTPUT
%       Out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] Weisheng Dong, Fazuo Fu, et. al.,"Hyperspectral image 
%           super-resolution via non-negative structured sparse 
%           representation", IEEE Trans. On Image Processing, vol. 25, 
%           no. 5, pp. 2337-2352, May 2015.
%--------------------------------------------------------------------------

rand('seed',0);

[rows1,cols1,bands1] = size(MSI);
[rows2,cols2,bands2] = size(HSI);

sz = [rows1,cols1]; 
kernel_type = "Uniform_blur";
sf = rows1/rows2;

par             =    NSSR_Parameters_setting( sf, kernel_type, sz );
HSI_2D          =    transpose(reshape(HSI,[rows2*cols2, bands2]));
MSI_2D          =    transpose(reshape(MSI,[rows1*cols1, bands1]));

[R,~] = estR(HSI,MSI);
for b = 1:bands1
    msi = reshape(MSI(:,:,b),rows1,cols1);
    msi = msi - R(b,end);
    msi(msi<0) = 0;
    MSI(:,:,b) = msi;
end
R = R(:,1:end-1);
param.K = bands2;
param.numThreads = 3;
param.iter = 300;
param.mode = 1;
param.lambda = 10e-9;
param.posD = 1; 
Phi = mexTrainDL(HSI_2D, param);
Phi_tilde = R*Phi;

par.P = Phi_tilde;
X = HSI_2D;
Y = MSI_2D;

%par.P           =    create_P();
%Y               =    par.P*HSI_2D;
%X               =    par.H(HSI_2D);

D               =    Nonnegative_DL( X, par );   
D0              =    par.P*D;
N               =    Comp_NLM_Matrix( Y, sz );   

Out         =    Nonnegative_SSR( D, D0, X, Y, N, par, NaN, sf, sz );
Out         =    reshape(transpose(Out),[rows1, cols1, bands2]);
