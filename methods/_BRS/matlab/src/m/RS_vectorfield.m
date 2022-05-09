function xi = RS_vectorfield(v, eps, gamma)
%RS_vectorfield Calculates the side information vector field
% 
% Input:  
%     v [matrix]
%         side information image        
%     eps [scalar] 
%         positive parameter         
%     gamma [scalar]
%         parameter in (0,1]
%         
% Output:
%     xi [matrix]
%         side information vector field
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
        
    g = RS_gradient(v); 
    n = sqrt(sum(g.^2, 3) + eps^2);
    % multiply: <g, v> * v
    xi = gamma * bsxfun(@times, 1./n, g); 
end