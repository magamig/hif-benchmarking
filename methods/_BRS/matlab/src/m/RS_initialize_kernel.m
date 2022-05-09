function k_init = RS_initialize_kernel(s_kernel)
%RS_initialize_kernel Defines the initial guess for the blurring kernel.
% 
% Input:
%     s_kernel
%         size of the blurring kernel
% 
% Output:
%     k_init
%         zero-mean gaussian kernel with approximately compact support
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

    assert(and(mod(s_kernel(1),2)~=0,mod(s_kernel(2),2)~=0),'Kernel size has to be odd!')
    l = (s_kernel-1)/2;   % kernel radius
    
    if (mod(l,2)==0)
            l = l+1; %l becomes odd
    end
    
    var = (min(l)-1)/6;   % with this variance the essential support of temp lies in [-l,l]^2
    temp = fspecial('gaussian', l,var);
    k_init = RS_pad_array(temp,s_kernel);
            
end