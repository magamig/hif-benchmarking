function ret = RS_downsampling_adjoint_invers(data,downsamplingFactor,rho,S,Sadj)
% RS_downsampling_adjoint_invers Computes the inverse of (Sadj o S + rho*I),
% where Sadj is the adjoint of S, I denotes the identity and o the
% composition operator.
% 
% Input:
%     data [matrix]
%         input data
%     downsamplingFactor [int]
%         factor determining the dimensions of the downsampled image
%     rho []
%         tbd
%     S [function handle]
%         downsampling operator
%     Sadj [function handle]
%         adjoint downsampling operator
%         
% Output: 
%     ret [matrix]
%         output data
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
    
    retInside = Sadj(S(data))/((rho^2 + s^2*rho*s^-4));
    retInside = ((s^2)*s^-4+(rho))*data/((rho^2 + s^2*rho*s^-4)) - retInside;
    ret = retInside;
end

