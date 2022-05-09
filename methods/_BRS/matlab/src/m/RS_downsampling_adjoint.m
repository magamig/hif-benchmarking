function dataUS = RS_downsampling_adjoint(data,upsamplingFactor,margin)
%RS_downsampling_adjoint Upsamples an image, adjoint operator of RS_downsampling.
% 
% Input:
%     data [matrix]
%         image to be upsampled
%     upsamplingFactor [int]
%          factor determining the dimensions of the upsampled image
%     margin [vector]
%         size of zero-padding margin
%         
% Output:
%     dataUS [matrix]
%         upsampled image
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
    
    s = upsamplingFactor;
    l = margin;
    sImage = s*size(data) + 2*l;        
    data = reshape(data, size(data));
    aux = ones(s);
    dataUS = kron(data,aux)/(s^2);
    dataUS = RS_pad_array(dataUS, sImage);
end