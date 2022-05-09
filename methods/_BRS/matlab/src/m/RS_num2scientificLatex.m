function s = RS_num2scientificLatex(x)
%RSnum2scientificLatex Converts to LaTex-comprehensible scientific number notation. 
%   Input:
%     x [double]
%         input number
%   Output:
%     s [string]
%         representation of x in scientific notation
%         that can be treated by LaTex
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

    if isnan(x)
        s = '';
    else
        s = [];
        if x < 0
            s = [s,'-'];
        end
        x = abs(x);
        s2 = num2str(x,'%1.2e');
%         s2 = strsplit(s2,'e');    
%         if str2num(s2{1}) ~= 1
%             s = [s,num2str(str2num(s2{1})),'\times'];
%         end    
%         s = [s,'10^{',num2str(str2num(s2{2})),'}'];    

        s2 = num2str(x,'%1.0e');
        s2 = strsplit(s2,'e');
%         if str2num(s2{1}) ~= 1
%             s = [s,num2str(str2num(s2{1})),'\times'];
%         end
        s = [s2{1},'e'];
        exponent = str2num(s2{2});
        if exponent >= 0
            s = [s,'+'];
        end
        s = [s, num2str(exponent)];
    end
end
