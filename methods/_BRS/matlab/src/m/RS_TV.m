function fun = RS_TV(u, lambda)
%RS_TV Computes the TV prior.
% 
% Input:
%     u [matrix]
%         image    
%     lambda [scalar]
%         regularization parameter
%         
% Output:
%     fun [scalar]
%         value of the TV prior
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
        
    if lambda > 0
        % calulate the gradient
        g = RS_gradient(u);    
        % and store its norm
        n = sqrt(sum(g.^2, 3));
        % sum up over domain
        fun = lambda * sum(n(:));
    else
        fun = 0;
    end
end
