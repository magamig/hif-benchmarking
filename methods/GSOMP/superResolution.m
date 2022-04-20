function [] = superResolution(param)
%% superResolution(paramters)
%   This function perfroms Hyperspectral image super resolution using
%   sparse spatio-spectral representation using the technique presented in
%   ECCV 2014 paper.
%   Please cite the following paper, if you use the code:
%   Akhtar, Naveed, Faisal Shafait, and Ajmal Mian. "Sparse Spatio-spectral Representation for Hyperspectral Image Super-Resolution." Computer Vision–ECCV 2014. Springer International Publishing, 2014. 63-78.
%   The notations below are according to the paper.
%% ========== Optional parameters ============
%   'spams' - Set it to 1 if SPAMS library*** is installed
%       Default: spams = 0 (i.e. no spams library installed)
%                Note that, the dictionary provided with the code is only for the Faces
%                image. In the case of no SPAMS, provide the right dictionary for correct results.
%   'L' - Number of atoms selected in each iteration of G-SOMP+
%       Default: L = 20
%   'gamma' - Residual decay parameter used by G-SOMP+
%       Default: gamma = 0.99
%   'k' - Number of dictionary atoms to be used in the learning process
%       Default: k = 300
%   'eta' - modeling error Equation (7) in the paper
%       Default: 10e-9
%   'HSI' - name of the image HS image file. The data file should be in the
%   working directory.
%       Default: 'Faces'
%
% ***SPAMS (SPArse Modeling Software) is an open source tool that can be easily searched on the internet.
%--------------------------------------------------------------
% Set the defaults for the parameters 
%--------------------------------------------------------------
spams = 0;
L = 20;
gamma = 0.99;
k = 300;
eta = 10e-9;
HSI = 'Faces';

%--------------------------------------------------------------
% Simulated Nikon D700 spectral response T = [R;G;B]
%--------------------------------------------------------------
%Note, the same matrix 'T' must be used in transforming the dictionary and the
%ground truth to create the high-res RGB image. (That matrix can be different from
%the one used below).
T = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004 0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0  
    0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003
    0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];

%--------------------------------------------------------------
% Read the parameters, if given
%--------------------------------------------------------------
if isfield(param,'spams')
    if param.spams == 0 || param.spams== 1
        spams = param.spams;
    else
        error('wrong value of the parameter spams')
    end
    if param.spams == 1
        disp('The demo will use the dictionary after learning it')
    end
end
if isfield(param, 'L')
    L = param.L;
end
if isfield(param, 'gamma')
    gamma = param.gamma;
end
if isfield(param, 'k');
    k = param.k;
end
if isfield(param, 'eta')
    eta = param.eta;
end
if isfield(param, 'HSI')
    HSI = param.HSI;
end

%--------------------------------------------------------------
% Read the input image and downsample it
%--------------------------------------------------------------
downsampling_scale = 32;
im_structure = load(HSI);
S = im_structure.im;           
[M,N,L] = size(S);
% If the image is not square, make it so.
min_dim = min(M,N);
dim = floor(min_dim/8)*8;
S = S(1:dim, 1:dim, :);
[Y_h] = downsample(S, downsampling_scale); %%%%%%%%%%%%%%%%%%%%%%%%%%%% replace Y_h by HSI
Y_h_bar = hyperConvert2D(Y_h);

%--------------------------------------------------------------
% Simulate high resolution image (i.e. RGB image)
%--------------------------------------------------------------
S_bar = hyperConvert2D(S);
Y = hyperConvert3D((T*S_bar), M, N, 3); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% replace Y by MSI

%--------------------------------------------------------------
% Learn the dictionary if SPAMS library is installed
% Otherwise load the learned ditionary for 'Faces'.
% Faces_Dictionry should only be used for the Faces image.
%--------------------------------------------------------------
if param.spams == 1
   %See the documentation of the spams library for the parameters
   param.K = k;
   param.numThreads = 3;
   param.iter = 300;
   param.mode = 1;
   param.lambda = eta;
   param.posD = 1; 
   Phi = mexTrainDL(Y_h_bar,param);
else
   struc = load('Faces_Dictionary');
   Phi = struc.Phi;
end
Phi_tilde = T*Phi;

%--------------------------------------------------------------
% Process image in patch by patch manner
%--------------------------------------------------------------
patchsize = 8;
total_patches = (dim/patchsize)^2;
Y_bar = convert3Dto2Dpatchwise( Y, patchsize );
S_bar_patchwise = convert3Dto2Dpatchwise(S, patchsize);

temp = Y_bar;
Sparse_code = [];

patch_no = 1;
tic
for i = 1:total_patches
    if i == patch_no
        disp(['Processing patch ' int2str(patch_no) ' to ' int2str(patch_no+50)])
        patch_no = patch_no + 50;
    end
    A = GSOMP_NN(Phi_tilde, temp(:,1:patchsize^2), L, gamma);
    Sparse_code = [Sparse_code A];
    temp(:,1:patchsize^2) = [];
end
timer = toc;
disp(['G-SOMP+ processing time....: ' int2str(timer) 'seconds']) 
S_bar_cap = Phi * Sparse_code;
RMSE = sqrt((norm((double(im2uint8(S_bar_patchwise)) - double(im2uint8(S_bar_cap))),2))^2/(M*N*L)) 
%The same metric has been used in evaluating the other appraoches in our
%paper. Please also evaluate those approaches separately, if a different
%definition of RMSE is used in your work.

%--------------------------------------------------------------
% Display images
%--------------------------------------------------------------
disp('Constructing the hyperspectral image from the 2D patches...')
[ estimated3D ] = convert2Dto3Dpatchwise( S_bar_cap, patchsize );

r = 7; g = 15; b = 21;
err460 = double(im2uint8(S(:,:,r)))- double(im2uint8(estimated3D(:,:,r)));
err540 = double(im2uint8(S(:,:,g)))- double(im2uint8(estimated3D(:,:,g)));
err620 = double(im2uint8(S(:,:,b)))- double(im2uint8(estimated3D(:,:,b)));

figure(1),subplot(3,3,1),imshow(Y_h(:,:,r)),title('460nm (Input)'),subplot(3,3,2),imshow(Y_h(:,:,g)),title('540nm (Input)'),subplot(3,3,3),imshow(Y_h(:,:,b)),title('620nm (Input)'), subplot(3,3,4),imshow(S(:,:,r)),title('460nm (Ground truth)'),subplot(3,3,5),imshow(S(:,:,g)),title('540nm (Ground truth)'),subplot(3,3,6),imshow(S(:,:,b)),title('620nm (Ground truth)'), subplot(3,3,7),imshow(estimated3D(:,:,r)),title('460nm (Estimate)'),subplot(3,3,8),imshow(estimated3D(:,:,g)),title('540nm (Estimate)'),subplot(3,3,9),imshow(estimated3D(:,:,b)),title('620nm (Estimate)')   
figure(2),subplot(1,3,1),imagesc(err460, [-20 20]),title('460nm (Error)'), subplot(1,3,2), imagesc(err540, [-20 20]),title('540nm (Error)'),subplot(1,3,3), imagesc(err620, [-20 20]),title('620nm (Error)')

end

function [down_im] = downsample(im, scale)
new_pixel_size = scale;
a = size(im,1)/new_pixel_size;
down_im = zeros(a,a,size(im,3));
m = zeros(a, size(im,1));
s = 1:new_pixel_size:size(im,1);
e = new_pixel_size:new_pixel_size:size(im,1);

for l = 1:size(im,3)
    ima(:,:) = im(:,:,l);
    temp1 = [];
    for i = 1:a
        t = sum(ima(s(i):e(i), :));
        temp1 = [temp1; t];
    end 
    temp2 = [];
    for i = 1:a 
        t = sum(temp1(:,s(i):e(i)),2);
        temp2 = [temp2 t];
    end
    temp2 = temp2./repmat(new_pixel_size^2,size(temp2,1),size(temp2,1));
    down_im(:,:,l) = temp2;
end
    
end

function [Image2D] = hyperConvert2D(Image3D)
if (ndims(Image3D) == 2)
    numBands = 1;
    [h, w] = size(Image3D);
else
    [h, w, numBands] = size(Image3D);
end
Image2D = reshape(Image3D, w*h, numBands).';
end

function [Image3D] = hyperConvert3D(Image2D, h, w, numBands)
[numBands, N] = size(Image2D);
if (1 == N)
    Image3D = reshape(Image2D, h, w);
else
    Image3D = reshape(Image2D.', h, w, numBands); 
end
end

function [ mat2D ] = convert3Dto2Dpatchwise( im3D, n )
[y, x, d] = size(im3D);
np = floor(x/n); 
s = 1:n:x;
e = n:n:x;
temp = zeros(y,n,d);    
mat2D = [];
for j = 1:np           
    temp = im3D(:,s(j):e(j),:);   
    temp2 = zeros(n,n,d);
    for k = 1:np
        temp2 = temp(s(k):e(k),:,:);  
        patch2D = reshape(temp2, n*n, d).';
        mat2D = [mat2D patch2D];
    end   
end
end

function [ Aa ] = GSOMP_NN( D,Y,L,gamma)
[m, n_p] = size(Y);
[m, k] = size(D);
Dn = normc(D);
R = Y;
S = [];
    while 1
        R0 = R;
        [v, indx] = sort(sum((Dn' * normc(R)),2), 'descend');
        j = indx(1:L,1);
        S = union(S, j');   
        Aa = [];
        aa = zeros(k,1);
        for i =1:n_p        
            a = lsqnonneg(D(:,S), Y(:,i));
            aa(S,1) = a;
            Aa = [Aa aa];
            R(:,i) = Y(:,i)- D(:,S)*a;
        end   
        if norm(R,2) > gamma*norm(R0,2)      
            break;
        end
    end
end

function [ mat3D ] = convert2Dto3Dpatchwise( im2D, n )
[l N] = size(im2D);
x = sqrt(N);
y = sqrt(N);
indx = 1:n:x;
indxend = n:n:x;
indy = 1:n:y;
indyend = n:n:y;
im3D = zeros(x, y, l);
temp = im2D;
for j = 1:length(indy)
   for i = 1:length(indx)
       im = temp(:,1:n^2);
       patch3D = hyperConvert3D(im, n, n, l);
       temp(:, 1:n^2) = [];
       im3D(indx(i):indxend(i), indy(j):indyend(j), :) = patch3D;
   end
end
   mat3D = im3D; 
end