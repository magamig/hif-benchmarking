function g = RS_gradient(u)
%RS_gradient Computes the circular gradient of scalar field u.
%
% Input:
%   u [matrix]
%       scalar field
%
% Output:
%   g [matrix]
%       gradient of u with periodic boundary conditions
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

    g = zeros([size(u) 2]);
    g(:,:,1) = circshift(u, -1, 1) - u;
    g(:,:,2) = circshift(u, -1, 2) - u; 
end