function out = MTF_GLP_wrapper(hs,ms)
%--------------------------------------------------------------------------
% This is a wrapper function that calls modulation transfer function (MTF) 
% based generalized Laplacian pyramid (GLP) [1-4] adapted for hyperspectral
% and multispectral data fusion via hypersharpening [5], to be used for the 
% comparisons in [6].
%
% USAGE
%       out = MTF_GLP_wrapper(hs,ms,ratio,mode)
%
% INPUT
%       hs  : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       ms  : MS image (rows1,cols1,bands1)
%       mode: mode for creating high resolution image (1: band selection; 
%       2: synthetic image via least squres; 3: synthetic image via
%       nonnegative least squares)
%
% OUTPUT
%       out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCE
%       [1] B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, "Context-
%           driven fusion of high spatial and spectral resolution images 
%           based on oversampled multiresolution analysis," IEEE 
%           Transactions on Geoscience and Remote Sensing, vol. 40, no. 10,
%           pp. 2300-2312, October 2002.
%       [2] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, 
%           "MTF-tailored multiscale fusion of high-resolution MS and Pan 
%           imagery," Photogrammetric Engineering and Remote Sensing, vol. 
%           72, no. 5, pp. 591-596, May 2006.
%       [3] G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. 
%           Chanussot, "Contrast and error-based fusion schemes for 
%           multispectral image pansharpening," IEEE Geoscience and Remote 
%           Sensing Letters, vol. 11, no. 5, pp. 930-934, May 2014.
%       [4] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. 
%           Garzelli, G. Licciardi, R. Restaino, and L. Wald, "A Critical 
%           Comparison Among Pansharpening Algorithms," IEEE Transaction on
%           Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565-2586, 
%           May 2015.
%       [5] M. Selva, B. Aiazzi, F. Butera, L. Chiarantini, and S. Baronti,
%           "Hypersharpening: A first approach on SIM-GA data," IEEE J. 
%           Sel. Topics Appl. Earth Observ. Remote Sens., vol. 8, no. 6, 
%           pp. 3008�3024, Jun. 2015.
%       [6] N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and 
%           multispectral data fusion: a comparative review of the recent 
%           literature," IEEE Geoscience and Remote Sensing Magazine, vol. 
%           5, no. 2, pp. 29-56, June 2017.
%--------------------------------------------------------------------------

ratio = size(ms,1)/size(hs,1);
mode = 2;

[rows1,cols1,bands1] = size(ms);
[rows2,cols2,bands2] = size(hs);

out = zeros(rows1,cols1,bands2);

low_res_ms = zeros(rows2,cols2,bands1);

for b = 1:bands1
    tmp = imresize(reshape(ms(:,:,b),rows1,cols1),[rows2 cols2],'bilinear');
    low_res_ms(:,:,b) = tmp;
end
%low_res_ms = gaussian_down_sample(ms,ratio);

X = zeros(bands1+1,bands2);
A = [reshape(low_res_ms,rows2*cols2,bands1) ones(rows2*cols2,1)];
switch mode
    case 1
        A = zeros(bands1,bands2);
        for i = 1:bands1
            tmp2 = reshape(low_res_ms(:,:,i),rows2,cols2);
            for j = 1:bands2
                tmp1 = reshape(hs(:,:,j),rows2,cols2);
                cc = corrcoef(tmp1(:),tmp2(:));
                A(i,j) = cc(1,2);
            end
        end
        [~,indices] = max(A,[],1);
    case 2
        B = (A'*A)\A'; 
        Y = reshape(hs,rows2*cols2,[]);
        X = B*Y;
    case 3
        X = nls_coef(reshape(hs,rows2*cols2,[]),A);    
end

A2 = [reshape(ms,rows1*cols1,bands1) ones(rows1*cols1,1)];
n = max(hs(:)).^0.5;

if mod(ratio,2) == 0
    posi = [ratio/2 ratio/2];
else
    posi = [round(ratio/2) round(ratio/2)];
end

for j = 1:size(hs,3)
    tmp = hs(:,:,j);
    switch mode 
        case 1
            pan = reshape(ms(:,:,indices(j)),rows1,cols1);
        case {2,3}
            pan = reshape(A2*X(:,j),rows1,cols1);
    end
    if mod(ratio,2) == 0
        MTF_GLP_tmp = (MTF_GLP(tmp,pan,'GaussKernel',ratio,1,n,posi)+MTF_GLP(tmp,pan,'GaussKernel',ratio,1,n,posi+1))/2;
    else
        MTF_GLP_tmp = MTF_GLP(tmp,pan,'GaussKernel',ratio,1,n,posi); 
    end
    out(:,:,j) = MTF_GLP_tmp;
end