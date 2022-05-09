function [x, param] = RS_process_data(x)
% RS_process_data Applies an affine normalization.
% 
% Input:
%     x [matrix]
%         image to be normalized
%         
% Output: 
%     x [matrix]
%         normalized image
%     param [struct]
%         parameters that determine the affine map
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
        
    b = min(x(:));
    a = max(x(:)) - b;
    
    x = (x - b)/a;
    
    param.a = a;
    param.b = b;
end    