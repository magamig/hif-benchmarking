function B_info = P_B_precomputations_square(A,X)

[m,n] = size(A);
[n,p] = size(X);

mn = m*n; mp = m*p; np = n*p;

% form the matrices of interest
XXt = X * X';

CA = zeros(mn,n);
for i = 1:n,
    CA(m*(i-1)+1:m*i,i) = A(:,i);
end

Ainv = inv(A);
AinvtAinv = Ainv' * Ainv; 

[Ux,Sx,Vx] = svd(XXt);
[Ua,Sa,Va] = svd(AinvtAinv);

mainBlockInv = .5 * ( kron(Ux,Ua) * diag((kron(diag(Sx),diag(Sa))+1).^-1) * kron(Ux',Ua') );
T = mainBlockInv * CA;
Q = -inv( CA' * T );

G_inv = zeros(mn+n,mn+n);
G_inv( 1:mn, 1:mn ) = mainBlockInv + T * Q * T';
G_inv( mn+1:mn+n, 1:mn ) = - Q * T';
G_inv( mn+1:mn+n, 1:mn ) = G_inv( 1:mn, mn+1:mn+n )';
G_inv( mn+1:mn+n, mn+1:mn+n ) = Q; 



B_info.A = A;
B_info.X = X; 
B_info.XXt = XXt;
B_info.Ainv = Ainv;
B_info.AinvtAinv = AinvtAinv;
B_info.CA = CA;
B_info.Ginv = G_inv; 