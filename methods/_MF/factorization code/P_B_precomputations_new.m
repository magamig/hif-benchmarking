function B_info = P_B_precomputations_new(A,X)

[m,n] = size(A);
[n,p] = size(X);

mn = m*n; mp = m*p; np = n*p;

% form the matrices of interest
XXt = X * X';
AAt = A * A';
AAt_inv = inv(AAt);

[UA,LambdaA,DC] = svd(AAt);
[UX,LambdaX,DC] = svd(XXt);

lambdaA = diag(LambdaA);
lambdaX = diag(LambdaX);

Gamma = repmat(lambdaA,1,n) ./ ( repmat(lambdaA,1,n) + repmat(lambdaX',m,1) );

Xi_inv = zeros(n,n);
UAt_A = UA' * A;
UXt   = UX';
for i = 1:n,
    for j = i:n,
        Xi_inv(i,j) = (UAt_A(:,i) .* UAt_A(:,j))' * Gamma * (UXt(:,i) .* UXt(:,j));
        Xi_inv(j,i) = Xi_inv(i,j);        
    end
end
Xi = inv(Xi_inv);

B_info.A = A;
B_info.X = X;
B_info.AAt_inv = AAt_inv;
B_info.Xi = Xi;
B_info.UX = UX;
B_info.UA = UA;
B_info.Gamma = Gamma; 
