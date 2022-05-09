function ind = RS_embedding_indices(kernelSize,imageSize)
%RS_embedding_indices Computes the indices of kernel elements after zero-padding.
% 
% Input:
%     kernelSize [vector]
%         number of rows and columns of the kernel
%     imageSize [vector]
%         number of rows and columns of the image
% 
% Output:
%     int [vector]
%         array indices of the kernel elements
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
        
    n = kernelSize(1); m = kernelSize(2);
    N = imageSize(1); M = imageSize(2);
    
    rect = reshape(1:n*m,n,m);
    pad = RS_pad_array(rect,[N,M]);
    [~,ind] = ismember(rect,pad);
    ind = Column(ind);

end