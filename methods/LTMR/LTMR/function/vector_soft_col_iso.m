function [Y1,Y2] = vector_soft_col_iso(X1,X2,tau)
%
%  computes the isotropic vector soft columnwise


NU = sqrt(sum(X1.^2)+sum(X2.^2));
A = max(0, NU-tau);
A = repmat((A./(A+tau)),size(X1,1),1);
Y1 = A.*X1;
Y2 = A.*X2;
