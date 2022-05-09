function Y = vector_soft_col(X,tau)
%
%  computes the vector soft columnwise


NU = sqrt(sum(X.^2));
A = max(0, NU-tau);
Y = repmat((A./(A+tau)),size(X,1),1).* X;
