function RS_save_kernel(kernel, file)
%RS_save_kernel Saves a blurring kernel to the disk. 
% A red haircross indicates the center whereas a blue haircross indicates 
% the barycenter of the kernel.
% 
% Input: 
%     kernel [matrix]
%         kernel to be saved          
%     file [string]
%         filename
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

    imwrite(VisualizeKernel(kernel), [file, '.png']);   

end

function ret = VisualizeKernel(kernel)

    assert(mod(size(kernel, 1), 2) ~= 0, 'Kernel must be odd sized')

    [s1, s2] = RS_kernel_barycenter(kernel);
    
    kk = repmat(kernel,[1, 1, 3]) - min(kernel(:));
    kk = kk ./ max(kk(:));
    l = (size(kk, 1) - 1)/2;
    
    r = l + 1; %part of red pixels on each side
    
    center = l+1;
    shift = [s1,s2]-center; 
        
    %numbers of blue pixels
    bL = r+shift(2);
    bR = r-shift(2);
    bU = r+shift(1);
    bD = r-shift(1);
           
    PosBlueDotsL = 1:2:bL;
    PosBlueDotsR = size(kk,1):-2:size(kk,1)-bR+1;
    PosBlueDotsU = 1:2:bU;
    PosBlueDotsD = size(kk,1):-2:size(kk,1)-bD+1;
    
    %red centered cross
    PosRedDots1 = 1:2:r;
    PosRedDots2 = size(kk,1):-2:size(kk,1)-r+1;
    
    kk(l+1,PosRedDots1,1) = .8;
    kk(l+1,PosRedDots2,1) = .8;
    kk(PosRedDots1,l+1,1) = .8;
    kk(PosRedDots2,l+1,1) = .8;

    for i = 2 : 3
        kk(l+1,PosRedDots1,i) = 0;
        kk(l+1,PosRedDots2,i) = 0;
        kk(PosRedDots1,l+1,i) = 0;
        kk(PosRedDots2,l+1,i) = 0;
    end
    
    %green off-centered cross
    if shift(1)~=0
        kk(s1,PosBlueDotsL,2) = .8;
        kk(s1,PosBlueDotsR,2) = .8;
    end 
    if shift(2)~=0 
        kk(PosBlueDotsU,s2,2) = .8;
        kk(PosBlueDotsD,s2,2) = .8;
    end 
        
    for i = [1, 3]
        if shift(1)~=0
            kk(s1,PosBlueDotsL,i) = 0;
            kk(s1,PosBlueDotsR,i) = 0;
        end 
        if shift(2)~=0 
            kk(PosBlueDotsU,s2,i) = 0;
            kk(PosBlueDotsD,s2,i) = 0;
        end 
    end
    
    ret = kk;
end

