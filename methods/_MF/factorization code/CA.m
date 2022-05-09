function B = CA(A,x)

[m,n] = size(A);
B = A .* repmat(x',m,1);