function Z = LSR2( X , lambda )

%--------------------------------------------------------------------------
% Copyright @ Can-Yi Lu, 2012
%--------------------------------------------------------------------------

% Input
% X             Data matrix, dim * num
% lambda        parameter, lambda>0


% Output the solution to the following problem:
% min ||X-XZ||_F^2+lambda||Z||_F^2

% Z             num * num


if nargin < 2
    lambda = 0.001 ;
end
[dim,num] = size(X) ;


% for i = 1 : num
%    X(:,i) = X(:,i) / norm(X(:,i)) ; 
% end


I = lambda * eye(num) ;
Z = (X'*X+I) \ X' * X ;