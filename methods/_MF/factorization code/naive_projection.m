function [Ap,Xp] = naive_projection(Y,A,X)

Ap = normalize_columns(A);
[V,S,U] = svd(Ap',0);
Xp = X + V * inv(S) * U' * (Y - Ap*X);

%Xp = X + Ap \ (Y - Ap * X);