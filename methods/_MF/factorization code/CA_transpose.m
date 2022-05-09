function z = CA_transpose(A,B)

[m,n] = size(A);

z = zeros(n,1);

for i = 1:n,
    z(i) = A(:,i)' * B(:,i);
end