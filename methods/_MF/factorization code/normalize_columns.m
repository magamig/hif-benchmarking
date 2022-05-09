function A_norm = normalize_columns(A)

[m,n] = size(A);
A_norm = A ./ repmat( sqrt(sum(A.*A,1)), m, 1);