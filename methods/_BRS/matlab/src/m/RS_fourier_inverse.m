function u = RS_fourier_inverse(Fu)
%RS_fourier_inverse Computes the inverse Fourier transform.
% 
% Input:
%     Fu [matrix]
%         image in frequency domain
%         
% Output: 
%     u [matrix]
%         image in spatial domain
%        
% See also:
%
% -------------------------------------------------------------------------
% Copyright 2017, L. Bungert, D. Coomes, M. J. Ehrhardt, M. A. Gilles, 
% J. Rasch, R. Reisenhofer, C.-B. Schoenlieb
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------

    RS_fft_count;
    
%     u = fftshift(fftn(ifftshift(Fu))) / sqrt(numel(Fu));
    u = fftshift(ifft2(ifftshift(Fu)));
end