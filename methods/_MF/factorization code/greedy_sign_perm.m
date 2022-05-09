function [P,sigma] = greedy_sign_perm(A,B)

% Given matrices A, B in \Re^{m \times n}, seek a permuation P of [n] such
% that the columns of A(P) are as similar to those of B as possible

[m,n] = size(A);

% compute (squared) distance matrix
D_plus = zeros(n,n);
D_minus = zeros(n,n);

for i = 1:n;
    for j = 1:n;
        D_plus(i,j) = (A(:,i) - B(:,j))' * (A(:,i)-B(:,j));
        D_minus(i,j) = (A(:,i) + B(:,j))' * (A(:,i) + B(:,j));
    end
end

I = 1:1:n;
J = 1:1:n;
P = zeros(1,n);
sigma = zeros(1,n);
curSign = 0;

for t = 1:n,   
    if ( min(min(D_plus(I,J))) <= min(min(D_minus(I,J))) ),
        [a,b]   = argmin(D_plus(I,J));  
        curSign = 1;
    else
        [a,b] = argmin(D_minus(I,J));
        curSign = -1;
    end               

    P(I(a)) = J(b);    
    sigma(I(a)) = curSign; 
    I = [I(1:a-1), I(a+1:end)];
    J = [J(1:b-1), J(b+1:end)];    
end