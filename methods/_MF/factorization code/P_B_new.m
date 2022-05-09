function y_out = P_B_new(z,B_info)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  P_B
%
%  Projects onto the orthogonal complement of the nullspace of the matrix
%
%    B = [  X' \tensor I   I \tensor A  ]
%        [  CA'            0            ]
%
%  Inputs:
%     z   - mn+np dimensional vector
%     B_info - struct, entries are:
%                B_info.A       - m x n matrix 
%                B_info.X       - n x p matrix
%                B_info.AAt_inv - m x m, (AA^*)^{-1} (precomputed)
%                B_info.XXt     - n x n matrix  XX^*, (precomputed)
%                B_info.Xi      - n x n, [ I_n - (XX^*) .* A^* (AA^*)^{-1} A ]
%                B_info.Phi     - mn x mn, 
%
%  Outputs:
%     y - mn+np dimensional output
%
%  Spring '10, John Wright - questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[m,n] = size(B_info.A);
[n,p] = size(B_info.X);

mn = m*n; mp = m*p; np = n*p;

% evaluate q = Bz
Q1 = reshape(z(1:mn),m,n) * B_info.X + B_info.A * reshape(z(mn+1:mn+np),n,p);    % 
Q2 = transpose(sum(B_info.A .* reshape(z(1:mn),m,n),1));                         % apply CA'

Y_tilde = B_info.AAt_inv * Q1;
[W,z_tilde] = apply_theta_new(Y_tilde * B_info.X', Q2,B_info);

Y = Y_tilde - B_info.AAt_inv * W * B_info.X;
z = Q2 - z_tilde;

% evaluate y = B* [Y,z]
y_out = [ vec( Y * B_info.X' + B_info.A .* repmat(z',m,1) ); vec( B_info.A' * Y ) ]; 

