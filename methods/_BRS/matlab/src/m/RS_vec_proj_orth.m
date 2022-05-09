function gxi = RS_vec_proj_orth(g, xi)
%RS_vec_proj_orth Projects onto the orthogonal complement of a vector field.
% 
% Input: 
%     g [matrix]
%         vectorfield to be projected       
%     xi [matrix]
%         reference vector field
%         
% Output:
%     gxi [matrix]
%         projection of g on the orthogonal complement of xi (assuming that 
%         xi has norm 1)
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

    % inner prod <g, xi>
    igxi = sum(bsxfun(@times, g, xi), 3);
    % multiply: <g, xi> * xi
    igxi_xi = bsxfun(@times, igxi, xi);
    % substract: g - <g, xi> * xi
    gxi = bsxfun(@minus, g, igxi_xi);
end