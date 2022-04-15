function Out = GSA_wrapper(hs,ms)
%--------------------------------------------------------------------------
% This is a wrapper function that calls the Gram-Schmidt Adaptive (GSA) 
% algorithm [1-2], to be used for the comparisons in [3].
%
% USAGE
%       Out = GSA_wrapper(hs,ms,ratio)
%
% INPUT
%       hs   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       ms   : MS image (rows1,cols1,bands1)
%       ratio: GSD ratio between HS and MS images
%
% OUTPUT
%       Out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCES
%       [1] B. Aiazzi, S. Baronti, and M. Selva, "Improving component 
%           substitution Pansharpening through multivariate regression of 
%           MS+Pan data," IEEE Transactions on Geoscience and Remote 
%           Sensing, vol. 45, no. 10, pp. 3230-3239, October 2007.
%       [2] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. 
%           Garzelli, G. Licciardi, R. Restaino, and L. Wald, "A Critical 
%           Comparison Among Pansharpening Algorithms," IEEE Transaction on
%           Geoscience and Remote Sensing, vol. 53, no. 2, pp. 2565-2586,
%           May 2015.
%       [3] N. Yokoya, C. Grohnfeldt, and J. Chanussot, "Hyperspectral and 
%           multispectral data fusion: a comparative review of the recent 
%           literature," IEEE Geoscience and Remote Sensing Magazine, vol. 
%           5, no. 2, pp. 29-56, June 2017.
%--------------------------------------------------------------------------

ratio = size(ms,1)/size(hs,1);

[rows1,cols1,bands1] = size(ms);
[rows2,cols2,bands2] = size(hs);

Out = zeros(rows1,cols1,bands2);

low_res_ms = zeros(rows2,cols2,bands1);

for b = 1:bands1
    tmp = imresize(reshape(ms(:,:,b),rows1,cols1),[rows2 cols2],'bilinear');
    low_res_ms(:,:,b) = tmp;
end

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

if mod(ratio,2) == 0
    posi = [ratio/2 ratio/2];
else
    posi = [round(ratio/2) round(ratio/2)];
end

for j = 1:bands1
    idx_tmp = find(indices==j);
    tmp = hs(:,:,idx_tmp);
    pan = reshape(ms(:,:,j),rows1,cols1);
    GSA_tmp = GSA(tmp,pan,posi);
    Out(:,:,idx_tmp) = GSA_tmp;
end