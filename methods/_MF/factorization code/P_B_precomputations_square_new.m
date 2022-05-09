function B_info = P_B_precomputations_square_new(A,X)

% Updated from P_B_precomputations_square --
%
%    Saves the main block implicitly

[m,n] = size(A);
[n,p] = size(X);

mn = m*n; mp = m*p; np = n*p;

if n ~= m,
    throw('P_B_precomputations_square_new - should be square m = n!');
end

% form the matrices of interest
XXt = X * X';

CA = zeros(mn,n);
for i = 1:n,
    CA(m*(i-1)+1:m*i,i) = A(:,i);
end

[Ux,Sx,Vx] = svd(XXt);

[Ua,Sa,Va] = svd(A);
Ainv = Va * diag(diag(Sa).^-1) * Ua';
AinvtAinv = Ainv' * Ainv; 

%[Ua,Sa,Va] = svd(AinvtAinv);

Nu = repmat(diag(Sa).^2,1,n);
Tau = repmat(diag(Sx)',m,1);
Gamma = .5 * Nu ./ ( Tau + Nu );

% mainBlockInv = .5 * ( kron(Ux,Ua) * diag((kron(diag(Sx),diag(Sa))+1).^-1) * kron(Ux',Ua') );

% mainBlockInv * v = 
% T = mainBlockInv * CA;

T = zeros(mn,n);
for i = 1:n,
    H = zeros(m,m);
    H(:,i) = A(:,i);
    T(:,i) = vec( Ua * ( Gamma .* ( Ua' * H * Ux ) ) * Ux' );       
end

Q = -inv( CA' * T );

%G_inv = zeros(mn+n,mn+n);
%G_inv( 1:mn, 1:mn ) = mainBlockInv + T * Q * T';
%G_inv( mn+1:mn+n, 1:mn ) = - Q * T';
%G_inv( mn+1:mn+n, 1:mn ) = G_inv( 1:mn, mn+1:mn+n )';
%G_inv( mn+1:mn+n, mn+1:mn+n ) = Q; 


B_info.A = A;
B_info.X = X; 
B_info.Ua = Ua;
B_info.Ux = Ux;
B_info.XXt = XXt;
B_info.Ainv = Ainv;
B_info.AinvtAinv = AinvtAinv;
B_info.CA = CA;
B_info.Q = Q;
B_info.T = T;
B_info.Gamma = Gamma;
%B_info.Ginv = G_inv; 