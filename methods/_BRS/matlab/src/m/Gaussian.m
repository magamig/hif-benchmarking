function [g, sigma] = Gaussian(FWHM, s_kernel, s_voxel)
%Gaussian Computes a Gaussian kernel 
% which can be used for instance for convolution.
%
% Input:
%   FWHM [scalar]
%       full width half maximum of the kernel
%   s_kernel [vector] 
%       size of the kernel
%   s_voxel [vector] 
%       size of the voxel
%
% Output:
%   g [matrix]
%       voxelized Gaussian kernel
%   sigma [scalar] 
%       standard deviation of the kernel
%
% See also:
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-10-14 --------------------------------------------------------------
% Matthias J. Ehrhardt
% CMIC, University College London, UK 
% matthias.ehrhardt.11@ucl.ac.uk
% http://www.cs.ucl.ac.uk/staff/ehrhardt/software.html
%
% -------------------------------------------------------------------------
% Copyright 2015 University College London
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%   http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% -------------------------------------------------------------------------

    if nargin < 1 || isempty(FWHM); FWHM = 5; end; % mm
    if nargin < 3 || isempty(s_voxel); s_voxel = [1 1]; end; % mm
    
    dim = min(numel(FWHM), numel(s_voxel));
    
    sigma = zeros(dim,1);
    for i = 1 : dim
        sigma(i) = FWHM(i) ./ (2 * sqrt(2 * log(2))) ./ s_voxel(i);
    end
    
    default_s_kernel = max(ceil(5*sigma), 3);
    default_s_kernel = default_s_kernel + mod(default_s_kernel+1,2);
    if nargin < 2 || isempty(s_kernel);
        s_kernel = default_s_kernel;
    else
        for k = 1 : length(s_kernel); if ~s_kernel(k); s_kernel(k) = default_s_kernel; end; end;
    end;
        
    switch dim
        case 1
            g = fspecial('Gaussian', [s_kernel(1), 1], sigma(1));
        
        case 2                       
            g1 = fspecial('Gaussian', [s_kernel(1), 1], sigma(1));
            g2 = fspecial('Gaussian', [s_kernel(2), 1], sigma(2));
            
            g = g1 * g2';
           
        case 3
            g1 = fspecial('Gaussian', [s_kernel(1), 1], sigma(1));
            g2 = fspecial('Gaussian', [s_kernel(2), 1], sigma(2));
            g3 = fspecial('Gaussian', [s_kernel(3), 1], sigma(3));
            
            g = zeros(s_kernel);
            g12 = g1 * g2';
            
            for i = 1 : s_kernel(3)
               g(:,:,i) = g12 * g3(i); 
            end
            
        otherwise
            error('Dimension %i not yet supported', dim);
    end        
end