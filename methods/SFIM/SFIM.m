function out = SFIM(hs,ms)
%--------------------------------------------------------------------------
% Smoothing filter-based intensity modulation (SFIM) adapted for
% hyperspectral and multispectral data fusion via hypersharpening
%
% USAGE
%       out = SFIM(hs,ms,mask)
%
% INPUT
%       hs  : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       ms  : MS image (rows1,cols1,bands1)
%       mode: mode for creating high resolution image (1: band selection; 
%       2: synthetic image via least squres; 3: synthetic image via
%       nonnegative least squares)
%       mask: (optional) Binary mask for processing (rows2,cols2)
%
% OUTPUT
%       out   : High-spatial-resolution HS image (rows1,cols1,bands2)
%
% REFERENCE
% [1] J. G. Liu, "Smoothing filter-based intensity modulation: a spectral
% preserve image fusion technique for improving spatial details," Int. J.
% Remote Sens. vol. 21, no. 18, pp. 3461-3472, Dec. 2000.
% [2] M. Selva, B. Aiazzi, F. Butera, L. Chiarantini, and S. Baronti, 
% "Hypersharpening: A first approach on SIM-GA data," IEEE J. Sel. Topics 
% Appl. Earth Observ. Remote Sens., vol. 8, no. 6, pp. 3008�3024, Jun.
% 2015.
%--------------------------------------------------------------------------

mode = 3;

[rows1,cols1,bands1] = size(ms);
[rows2,cols2,bands2] = size(hs);

out = zeros(rows1,cols1,bands2);

w = rows1/rows2;

% masking mode
if mode == 3
    masking = 0;
elseif mode == 4
    masking = 1;
    mask2 = imresize(mask,w,'nearest');
else
    disp('Please check the usage of SFIM.m');
end

low_res_ms = zeros(rows2,cols2,bands1);

for b = 1:bands1
    tmp = imresize(reshape(ms(:,:,b),rows1,cols1),[rows2 cols2],'bilinear');
    low_res_ms(:,:,b) = tmp;
end
% original implementation (mean filtering)
%low_res_ms = permute(reshape(mean(reshape(permute(reshape(mean(reshape(ms,w,[],bands1),1),rows2,[],bands1),[2 1 3]),w,[],bands1),1),cols2,rows2,bands1),[2 1 3]);

% find coefficients
X = zeros(bands1+1,bands2);
A = [reshape(low_res_ms,rows2*cols2,bands1) ones(rows2*cols2,1)];
switch mode
    case 1
        A = zeros(bands1,bands2);
        for i = 1:bands1
            tmp2 = reshape(low_res_ms(:,:,i),rows2,cols2);
            if masking == 1
                tmp2 = tmp2(mask);
            end
            for j = 1:bands2
                tmp1 = reshape(hs(:,:,j),rows2,cols2);
                if masking == 1
                    tmp1 = tmp1(mask);
                end
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
for i = 1:bands2
    tmp1 = reshape(hs(:,:,i),rows2,cols2);
    tmp1 = imresize(tmp1,w,'nearest');
    switch mode 
        case 1
            tmp2 = reshape(ms(:,:,indices(i)),rows1,cols1);
            tmp3 = reshape(low_res_ms(:,:,indices(i)),rows2,cols2);
        case {2,3}
            tmp2 = reshape(A2*X(:,i),rows1,cols1);
            tmp3 = reshape(A*X(:,i),rows2,cols2);
    end
    tmp3 = imresize(tmp3,w,'nearest');
    tmp4 = tmp1;
    if masking == 1
        tmp4(mask2) = tmp2(mask2).*tmp1(mask2)./tmp3(mask2);
    else
        tmp4 = tmp2.*tmp1./tmp3;
    end
    out(:,:,i) = tmp4;
end