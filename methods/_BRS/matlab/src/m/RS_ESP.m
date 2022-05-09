function pk = RS_ESP(k)
%RS_ESP Performs the Euclidean Simplex Projection [1].
%
% [1] Duchi, John, et al. "Efficient projections onto the l 1-ball for learning in high dimensions." 
% Proceedings of the 25th international conference on Machine learning. ACM, 2008.
% 
% Input:
%     k [matrix]
%         un-normalized kernel 
%     
% Output:
%     pk [matrix]
%         projection of k onto the unit simplex
%        
% See also:
%
% -------------------------------------------------------------------------
% Copyright 2017, L. Bungert, D. Coomes, M. J. Ehrhardt, J. Rasch, 
% R. Reisenhofer, C.-B. Schoenlieb
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

    k = Column(k);
    n = length(k);
    mu = sort(k, 'descend');
    crit = mu-(1./(1:n)') .* (cumsum(mu)-1);
    rho = find(crit>0, 1, 'last');
    theta = 1/rho*(sum(mu(1:rho))-1);
    
    pk = max(k-theta,0);
    
end
