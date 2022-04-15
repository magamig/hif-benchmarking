function [ V, R, B ] = sen_resp_est( Yhim, Ymim, downsamp_factor, intersection, ...
                            contiguous, p, lambda_R, lambda_B, hsize_h, hsize_w, shift, blur_center )
%sen_resp_est - Sensors' response estimation
% 
%   Estimates the relative spatial and spectral responses of the 
%   hyperspectral and multispectral/panchromatic sensors. 
% 
% [ V, R, B ] = sen_resp_est( Yhim, Ymim, downsamp_factor, intersection,
%                             contiguous, p, lambda_R, lambda_B )
% 
% Input: 
% Yhim: HS image, 
% Ymim: MS image, 
% downsamp_factor: downsampling factor, 
% intersection: cell array with the spectral coverage of both sensors, 
%     i.e., which HS band corresponds to each MS band
%     e.g.: MS band 1 - HS bands 1,2,3,4,5
%           MS band 2 - HS bands 6,9 (7 and 8 were removed for some reason)
%           MS band 3 - ..., 
% contiguous: if some bands were removed from the original datacube, this 
%     cellarray is used to keep track of the bands that are contiguous in 
%     the data cube but are not in the sensor. If there are no
%     removed bands, it can be set to be the same as 'intersection'.
% p: corresponds to variable L_s in [1]; number of endmembers in VCA /
%     number of non-truncated singular vectors,
% lambda_R: regularization parameter lambda_R, 
% lambda_B: regularization parameter lambda_B,
% hsize_h, hsize_w: blur's support
% shift: 'phase' parameter in MATLAB's 'upsample' function
% blur_center: used to regulate the position of the blur wrt to the
% observed MS image
% 
% Output: 
% V: subspace estimated with SVD
% R: relative spectral resppnse; it has the structure discussed in [1]
% B: relative spatial response; B is a 2D matrix corresponding
%   to the blur kernel and not a BCCB matrix as in [1]. This BCCB
%   matrix is never explicitly computed since we use FFTs.
% 
%   For more details on this, see Section IV of
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A convex formulation for hyperspectral image superresolution 
%        via subspace-based regularization,?IEEE Trans. Geosci. Remote 
%        Sens., to be publised.
 
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% Can be obtained online from: https://github.com/alfaiate/HySure
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2014 Miguel Simoes, Jose Bioucas-Dias, Luis B. Almeida 
% and Jocelyn Chanussot
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% 
% % % % % % % % % % % % % 
% 
% This funtion has six steps:
% 
% I. Blur Ym with a strong blur. The dimension of this blur can be changed:
%
% Blur's support: [ly*2+1 lx*2+1]
lx = 4;
ly = 4;
% 
% II. Blur Yh with a correspondingly scaled blur.
% 
% III. Estimate R on the blurred data.
% 
% IV. Data denoising. We do this before estimating B because we are
% no longer working with blurred versions of the images, which are already
% somewhat ''denoised''.
% 
% V. Estimate B on the original observed data. The support of the blur to
% be estimated can be specified:
% 
% VI. Normalize B to unit DC gain.
%  
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% I. Blur Ym with a strong blur.                                        %
% ------------------------------                                        %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

[nl, nc, nb] = size(Ymim);
% Blur operator
middlel = round((nl+1)/2);
middlec = round((nc+1)/2);
% Blur matrix
Bm = zeros(nl, nc);
Bm(middlel-ly:middlel+ly, middlec-lx:middlec+lx) = fspecial('average', [ly*2+1 lx*2+1]);
% Circularly center
Bm = ifftshift(Bm);
% Normalize
Bm = Bm/sum(sum(Bm));
% Fourier transform of the filters
FBm = fft2(Bm);
% Blur Ym
Ym = im2mat(Ymim);
Ymb = ConvC(Ym, FBm, nl);
% Masked version of Ym (to use when estimating the spatial blur)
Ymbim = mat2im(Ymb, nl);
Ymbim_down = downsamp_HS(Ymbim, downsamp_factor, shift);
Ymb_down = im2mat(Ymbim_down);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% II. Blur Yh with a correspondingly scaled blur.                       %
% -----------------------------------------------                       %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

[nlh, nch, L] = size(Yhim);
% Correspondingly scaled blur's support: [ly*2+1 lx*2+1]
lx = round(lx/downsamp_factor);
ly = round(ly/downsamp_factor);
% Blur operator
middlelh=round((nlh+1)/2);
middlech=round((nch+1)/2);
% Blur matrix
Bh=zeros(nlh, nch);
Bh(middlelh-ly:middlelh+ly, middlech-lx:middlech+lx) = fspecial('average', [ly*2+1 lx*2+1]);
% Circularly center
Bh = ifftshift(Bh);
% Normalize
Bh = Bh/sum(sum(Bh));
% Fourier transform of the filters
FBh = fft2(Bh);
% Blur Yh
Yh = im2mat(Yhim);
Yhb = ConvC(Yh, FBh, nlh);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% III. Estimate R on the blurred data.                                  %
% ------------------------------------                                  %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
 
R = zeros(nb, L);
% The estimate is made for each band, according to (15) and (16) in [1].
for idx_Lm = 1:nb   
    no_hs_bands =  length(intersection{idx_Lm});
    % Build a differences matrix
    col_aux = zeros(1, no_hs_bands-1);
    col_aux(1) = 1;
    row_aux = zeros(1, no_hs_bands);
    row_aux(1) = 1;
    row_aux(2) = -1;
    D = toeplitz(col_aux, row_aux);
%   When constructing the differences matrix D for the regularizer, 
%   non-contiguous bands should not be subtracted (this happens if they 
%   were removed due to being too noisy, for example). We find the 
%   bands in this situation and then elimate them from D.
    if (sum(diff(contiguous{idx_Lm})~=1))
        D(diff(contiguous{idx_Lm})~=1,:) = zeros(sum(diff(contiguous{idx_Lm})~=1), no_hs_bands);
    end
    DDt = D'*D;
    to_inv = Yhb(intersection{idx_Lm}, :)*Yhb(intersection{idx_Lm}, :)' + lambda_R*DDt;
    r = to_inv\(Yhb(intersection{idx_Lm}, :)*Ymb_down(idx_Lm, :)');
    R(idx_Lm, intersection{idx_Lm}) = r';
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% IV. Data denoising.                                                   %
% -------------------                                                   %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % %
% We will work with image Yhim in a matrix with the same size as Ymim. The
% result is a matrix filled with zeros. We do this for computational 
% convenience and the end result is the same. We also explicity form a
% subsampling mask that has the same effect has matrix M in [1].
mask = zeros(nl, nc);
mask(shift+1:downsamp_factor:nl, shift+1:downsamp_factor:nc) = 1;
% Subsampling mask in image format
maskim = repmat(mask, [1, 1, p]);
% Yhim with the same size as Ym (but with zeros)
Yhim_up = upsamp_HS(Yhim, downsamp_factor, nl, nc, shift);
Yh_up = im2mat(Yhim_up);

% Estimate the HS subspace with SVD to remove noise
[V,~] = svd(Yh);
V = V(:, 1:p);
% Project Yh on the subspace to remove noise
Yh_up = (V*V')*Yh_up;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% V. Estimate B on the original observed data.                          %
% ---------------------------------------------                         %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
        
% Differences matrices
col_aux = zeros(1, hsize_w-1);
col_aux(1) = 1;
row_aux = zeros(1, hsize_h);
row_aux(1) = 1;
row_aux(2) = -1;
Dv = toeplitz(col_aux, row_aux);
Dh = Dv';

% BTTB matrix
Ah = kron(Dh', eye(hsize_w));
ATAh = Ah'*Ah;

Av = kron(eye(hsize_h), Dv);
ATAv = Av'*Av;

% The estimate is made according to (14) from [1].
ryh = R*Yh_up;
ryhim = mat2im(ryh, nl);
ymymt = zeros((hsize_h)*(hsize_w), (hsize_h)*(hsize_w));
rtyhymt = zeros((hsize_h)*(hsize_w), 1);
for idx_L = 1:nb
    for idx_h = floor((hsize_h-1)/2) + 1:nl - floor((hsize_h-1)/2) - 1
        for idx_w = floor((hsize_w-1)/2) + 1:nc - floor((hsize_w-1)/2) - 1
%           Select only the pixels that can have the full blur's support 
%           arround them (ignore edges)
            if maskim(idx_h, idx_w, 1) == 1
                if (mod(hsize_h, 2) == 0) && (mod(hsize_w, 2) == 0)
                    patch = Ymim(idx_h - floor((hsize_h-1)/2):idx_h + floor((hsize_h-1)/2) + 1, idx_w -  floor((hsize_w-1)/2):idx_w +  floor((hsize_w-1)/2) + 1, idx_L);
                elseif (mod(hsize_h, 2) ~= 0) && (mod(hsize_w, 2) == 0)
                    patch = Ymim(idx_h - floor((hsize_h-1)/2):idx_h + floor((hsize_h-1)/2), idx_w -  floor((hsize_w-1)/2):idx_w +  floor((hsize_w-1)/2) + 1, idx_L);
                elseif (mod(hsize_h, 2) == 0) && (mod(hsize_w, 2) ~= 0)
                    patch = Ymim(idx_h - floor((hsize_h-1)/2):idx_h + floor((hsize_h-1)/2) + 1, idx_w -  floor((hsize_w-1)/2):idx_w +  floor((hsize_w-1)/2), idx_L);
                else
                    patch = Ymim(idx_h - floor((hsize_h-1)/2):idx_h + floor((hsize_h-1)/2), idx_w -  floor((hsize_w-1)/2):idx_w +  floor((hsize_w-1)/2), idx_L);
                end
                ymymt = ymymt + patch(:)*patch(:)';
                rtyhymt = rtyhymt + ryhim(idx_h, idx_w, idx_L)*patch(:);
            end
        end
    end
end

b_vec = (ymymt + lambda_B*(ATAh+ATAv))\rtyhymt;
% The blur is given by
b_vecim = reshape(b_vec, hsize_h, hsize_w);
% In a matrix with the same size as Ymim...
B = zeros(nl, nc);
if (mod(hsize_h, 2) == 0) && (mod(hsize_w, 2) == 0)
    B(middlel-floor((hsize_h-1)/2) - blur_center:middlel+floor((hsize_h-1)/2) + 1 - blur_center,middlec-floor((hsize_w-1)/2) - blur_center:middlec+floor((hsize_w-1)/2) + 1 - blur_center) = b_vecim;    
elseif (mod(hsize_h, 2) ~= 0) && (mod(hsize_w, 2) == 0)
    B(middlel-floor((hsize_h-1)/2) - blur_center:middlel+floor((hsize_h-1)/2) - blur_center,middlec-floor((hsize_w-1)/2) - 1 - blur_center:middlec+floor((hsize_w-1)/2) - blur_center) = b_vecim;
elseif (mod(hsize_h, 2) == 0) && (mod(hsize_w, 2) ~= 0)
    B(middlel-floor((hsize_h-1)/2) - 1 - blur_center:middlel+floor((hsize_h-1)/2) - blur_center,middlec-floor((hsize_w-1)/2) - blur_center:middlec+floor((hsize_w-1)/2) - blur_center) = b_vecim;
else
    B(middlel-floor((hsize_h-1)/2) - blur_center:middlel+floor((hsize_h-1)/2) - blur_center,middlec-floor((hsize_w-1)/2) - blur_center:middlec+floor((hsize_w-1)/2 - blur_center)) = b_vecim;
end
% Circularly center B
B = ifftshift(B);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% VI. Normalize B to unit DC gain.                                      %
% -------------------------------                                       %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Normalize B
B_vol = sum(B(:));
B = B/B_vol;
% Normalize R (because B was normalized)
R = R/B_vol;
end
