function [s1,s2] = RS_kernel_barycenter(k)
%RS_kernel_barycenter Computes the barycenter of a normalized blurring kernel 
% by evaluating the barycentric integral.
%
% Input: 
%     k [matrix]
%         normalized blurring kernel, i.e., sum(sum(k)) = 1
%         
% Output: 
%     s1,s2 [int] 
%         row and column indices of the approximate barycenter of k
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

    k = k-min(k(:));    % subtract the minimum value for better visualization
    k = k / sum(Column(k)); % enforce normalization
    len = size(k,1);
    x1 = double(repmat((1:len)',[1,len]));
    x2 = double(repmat(1:len,[len,1]));
    s1 = sum(Column(x1.*k));
    s2 = sum(Column(x2.*k));
    s1 = round(s1);
    s2 = round(s2);
    
end