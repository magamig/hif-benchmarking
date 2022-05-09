function I = columnwise_uniform_support_pattern(n,p,k)

I = zeros(n,p);

for j = 1:p,
    sigma = randperm(n);
    I(sigma(1:k),j) = 1;
end