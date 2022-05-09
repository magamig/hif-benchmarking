function dataDS = RS_downsampling(data,downsamplingFactor,margin)
% RS_downsampling Downsamples an image using an integrated sensor.
% 
% Input:  
%     data [matrix]
%         image to be downsampled
%     downsamplingFactor [int]
%         factor determining the dimensions of the downsampled image
%     margin [vector]
%         size of the cut-off margin
%         
% Output:
%     dataDS [matrix]
%         downsampled image
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
    
    s = downsamplingFactor;
    n = size(data);
    l = margin;
    sNew = (n-2.*l)./s;
    if abs(sNew(1) - round(sNew(1))) > 1e-6
        error('Something went wrong here. Dimensions mismatch.')
    end    
    dataDS = imresize(data((l+1):(end-l),(l+1):(end-l)),sNew,'box');   
end
