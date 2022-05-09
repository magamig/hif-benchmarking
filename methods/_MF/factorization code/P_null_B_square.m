function y_out = P_null_B_square(z,B_info)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  P_null_B_square
%
%  Projects onto the nullspace of the matrix
%
%    B = [  X' \tensor I   I \tensor A  ]
%        [  CA'            0            ]
%
%  Inputs:
%     z   - mn+np dimensional vector
%
%  Outputs:
%     y - mn+np dimensional output
%
%  This function uses the assumption that the matrix A is square and
%  invertible to significantly speed up the projection onto the nullspace
%  of the constraint matrix B. Should be useful for large-scale problems. 
%
%  Winter '10, John Wright - questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract the components (A,X) from z. 

[m,n] = size(B_info.A);
[n,p] = size(B_info.X);

mn = m*n; mp = m*p; np = n*p;

y_out = zeros(mn+np,1);

% RHS for KKT conditions
r = zeros(mn+n,1);
r(1:mn) = 2 * vec( - B_info.Ainv' * ( reshape(z(mn+1:end),n,p) * B_info.X')) + 2 * z(1:mn);

% solve KKT conditions for DeltaA part and Lagrange multiplier vector
% Lambda
h = B_info.Ginv * r;

y_out(1:mn)     = h(1:mn);
y_out(mn+1:end) = vec( ( -B_info.Ainv * reshape(y_out(1:mn),m,n) ) * B_info.X );

