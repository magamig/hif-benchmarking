function B = prune_library(A,min_angle)
%        B = prune_library(A,min_angle)
%
% remove columns from A such that the minimum angle between the columns
% of B in no smaller than max_angle
%
% Author: Jose Bioucas Dias. June, 2011
%

[L,m] = size(A);  % L = number of bands; m = number of materilas

%normalize A
nA = sqrt(sum(A.^2));
A_norm = A./repmat(nA,L,1);

% compute angles
angles = abs(acos(A_norm'*A_norm))*180/pi;


% discard vectors with angles less than min_angle
index = 1;
for i=1:m
   if angles(i,i) ~= inf
       B(:,index) = A(:,i);
       angles(:,angles(i,:) < min_angle ) = inf;
       index = index + 1;
   end

end





