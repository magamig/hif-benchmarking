function Zimhat = HySure_wrapper( HS, PAN )
%HySure_wrapper This is a wrapper function that calls HySure [1], to be
% used for the comparisons in [2]. The original source code, together with
% other datasets, can be obtained from https://github.com/alfaiate/HySure
% See the file README for more information.
% 
% Zimhat = HySure_wrapper( HS, PAN, downsamp_factor, overlap )
% 
% Input: 
% HS: HS image, 
% PAN: PAN image, 
% downsamp_factor: downsampling factor,
% overlap: array vector with the spectral coverage of both sensors, 
%     i.e., which HS bands are covered by the PAN band - optional
% 
% Output: 
% Zimhat: estimated image with high spatial and spectral resolution
%
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        �A convex formulation for hyperspectral image superresolution via 
%        subspace-based regularization,?IEEE Trans. Geosci. Remote Sens.,
%        to be publised.
% 
%   [2] Laetitia Loncan, Luis B. Almeida, Jose M. Bioucas-Dias, Xavier Briottet, 
%        Jocelyn Chanussot, Nicolas Dobigeon, Sophie Fabre, Wenzhi Liao, 
%        Giorgio A. Licciardi, Miguel Simoes, Jean-Yves Tourneret, 
%        Miguel A. Veganzones, Gemine Vivone, Qi Wei and Naoto Yokoya, 
%        "Introducing hyperspectral pansharpening," Geoscience and Remote Sensing
%        Magazine, 2015.

bands1 = size(PAN,3);
bands2 = size(HS,3);
overlap = repmat(1:bands2,[bands1 1]); 
downsamp_factor = size(PAN,1)/size(HS,1);
p = 30;
shift = round(downsamp_factor/2)-1;

% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2015 Miguel Simoes
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
% This function follows two steps. 
% 
% I. It starts by estimating the spectral and spatial response of the sensors.
% The regularization parameters can be adjusted here:
lambda_R = 1e1;
lambda_B = 1e1;
% For the denoising with SVD, we need to specify the number of bands we
% want to keep
%p = 30; % Corresponds to variable L_s in [1]; number of endmembers in VCA /
% number of non-truncated singular vectors
%
% 
basis_type = 'VCA';
lambda_phi = 1e-3;
lambda_m = 1;
% 
ms_bands = size(PAN,3); % Only panchromatic (HySure works for multispectral images 
%              as well, i.e., with more than one band)

% Normalize
max_HS = max(max(max(HS)));
Yhim = HS./max_HS;
Ymim = PAN./max_HS;

% Define the spectral coverage of both sensors, i.e., which HS band
% corresponds to each MS band
% e.g.: MS band 1 - HS bands 1,2,3,4,5
%       MS band 2 - HS bands 6,7,8,9
%       MS band 3 - ...
% Now imagine that there are some bands that are very noisy. We remove them
% from the hyperspectral data cube and build a vector with the number of 
% the bands there were not removed (we call this vector 'non_del_bands').
% For example, if we removed bands 3 and 4, non_del_bands = [1,2,5,6,...]
% We now define a cellarray, called 'intersection',  with length(ms_bands) 
% cells. Each cell corresponds to a multispectral band and will have a 
% vector with the number of the hyperspectral bands that are covered by it.
% Since we removed some bands as well, we need to keep track of the bands
% that are contiguous in the data cube but are not in the sensor.
% We call this other cellarray 'contiguous'. If there are no
% removed bands, it can be set to be the same as 'intersection'.
intersection = cell(1,length(ms_bands));
%if nargin == 3
%    intersection{1} = 1:size(HS, 3);
%elseif nargin == 4
for i = 1:ms_bands
    intersection{i} = overlap(i,:);
end
%else
%    disp('Please check the usage of HySure_wrapper');
%end
contiguous = intersection;
% Blur's support: [hsize_h hsize_w]
hsize_h = 2*downsamp_factor-1;
hsize_w = 2*downsamp_factor-1;
%shift = 1; % 'phase' parameter in MATLAB's 'upsample' function
blur_center = mod(downsamp_factor+1,2); % to center the blur kernel according to the simluated data
[V, R_est, B_est] = sen_resp_est(Yhim, Ymim, downsamp_factor, intersection, contiguous, p, lambda_R, lambda_B, hsize_h, hsize_w, shift, blur_center);
%%%%%%%%%%%%%%%%%%%%%%%%%%%% modification for HSMSFusionToolbox starts here
[R_est,~] = estR(Yhim,Ymim);
for b = 1:size(Ymim,3)
    msi = reshape(Ymim(:,:,b),size(Ymim,1),size(Ymim,2));
    msi = msi - R_est(b,end);
    msi(msi<0) = 0;
    Ymim(:,:,b) = msi;
end
R_est = R_est(:,1:end-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%% modification for HSMSFusionToolbox ends here
% II. The data fusion algorithm is then called using the estimated responses
% and the observed data. 

Zimhat = data_fusion(Yhim, Ymim, downsamp_factor, R_est, B_est, p, basis_type, lambda_phi, lambda_m, shift);

% Denoise the data again with V - optional
Zhat = im2mat(Zimhat);
Zhat_denoised = (V*V')*Zhat;
% In image form
Zimhat = mat2im(Zhat_denoised, size(PAN, 1));

% deNormalize
Zimhat = Zimhat.*max_HS;
end
