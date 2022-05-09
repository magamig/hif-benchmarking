function data_fid = RS_data_fidelity(A_u_k, data)
%RS_data_fidelity Computes the data fidelity.
%
% Input:
%   A_u_k [matrix]
%       application of forward operator to u and k    
%   data [matrix]
%       data image
%
% Output:
%   data_fid [scalar]
%       value of data fidelity
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

    res = A_u_k - data;      
    data_fid = 0.5 * sum(res(:).^2); % 
end    