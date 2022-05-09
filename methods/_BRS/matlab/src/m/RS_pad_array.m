function paddedArray = RS_pad_array(array, newSize)
%RS_pad_array Zero-pads an array to a certain size.
%
% Input:
%   array [matrix]
%		array to be padded
%	newSize [vector]
%		vector specifying the dimensions of the output
%
% Output:
%	paddedArray [matrix]
%		zero-padded array
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

    padSizes = zeros(1,length(newSize));
    S.type = '()';
    S.subs = cell(1,length(newSize));

    for k = 1:length(newSize)
        currSize = size(array,k);
        sizeDiff = newSize(k)-currSize;
        if mod(sizeDiff,2) == 0
            padSizes(k) = sizeDiff/2;    
            S.subs{k} = ':';
        else
            padSizes(k) = ceil(sizeDiff/2);
            if mod(currSize,2) == 0
                S.subs{k} = 2:(newSize(k)+1);
            else
                S.subs{k} = 1:(newSize(k));
            end
        end    
    end

    paddedArray = padarray(array,padSizes);
    paddedArray = subsref(paddedArray,S);


end
