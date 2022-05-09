function [psnr,rmse, ergas, sam, uiqi,ssim,DD,CCS] = quality_assessment(ground_truth, estimated, ignore_edges, ratio_ergas)
%quality_assessment - Computes a number of quality indices from the remote sensing literature,
%    namely the RMSE, ERGAS, SAM and UIQI indices. UIQI is computed using 
%    the code by Wang from "A Universal Image Quality Index" (Wang, Bovik).
% 
% ground_truth - the original image (3D image), 
% estimated - the estimated image, 
% ignore_edges - when using circular convolution (FFTs), the borders will 
%    probably be wrong, thus we ignore them, 
% ratio_ergas - parameter required to compute ERGAS = h/l, where h - linear
%    spatial resolution of pixels of the high resolution image, l - linear
%    spatial resolution of pixels of the low resolution image (e.g., 1/4)
% 
%   For more details on this, see Section V.B. of
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A convex formulation for hyperspectral image superresolution 
%        via subspace-based regularization,?IEEE Trans. Geosci. Remote 
%        Sens., to be publised.

% % % % % % % % % % % % % 
% 
% Author: Miguel Simoes
% 
% Version: 1
% 
% Can be obtained online from: https://github.com/alfaiate/HySure
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
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Ignore borders
y = ground_truth(ignore_edges+1:end-ignore_edges, ignore_edges+1:end-ignore_edges, :);
x = estimated(ignore_edges+1:end-ignore_edges, ignore_edges+1:end-ignore_edges, :);

% Size, bands, samples 
sz_x = size(x);
n_bands = sz_x(3);
n_samples = sz_x(1)*sz_x(2);

% RMSE
aux = sum(sum((x - y).^2, 1), 2)/n_samples;
rmse_per_band = sqrt(aux);
rmse = sqrt(sum(aux, 3)/n_bands);

% ERGAS
mean_y = sum(sum(y, 1), 2)/n_samples;
ergas = 100*ratio_ergas*sqrt(sum((rmse_per_band ./ mean_y).^2)/n_bands);

% SAM
sam= SpectAngMapper( ground_truth, estimated );
sam=sam*180/pi;
% num = sum(x .* y, 3);
% den = sqrt(sum(x.^2, 3) .* sum(y.^2, 3));
% sam = sum(sum(acosd(num ./ den)))/(n_samples);

% UIQI - calls the method described in "A Universal Image Quality Index"
% by Zhou Wang and Alan C. Bovik
q_band = zeros(1, n_bands);
for idx1=1:n_bands
    q_band(idx1)=img_qi(ground_truth(:,:,idx1), estimated(:,:,idx1), 32);
end
uiqi = mean(q_band);
ssim=cal_ssim(ground_truth, estimated,0,0);
DD=norm(ground_truth(:)-estimated(:),1)/numel(ground_truth);
CCS = CC(ground_truth,estimated);
CCS=mean(CCS);
psnr=csnr(ground_truth, estimated,0,0);