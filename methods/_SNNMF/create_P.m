function [P] = create_P(dsf,roi)
%CREATE_P Summary of this function goes here
%   P permutes G into a block-diagonal matrix
%       i.e. PG = [block diagonal]

w = dsf*(roi(2) - roi(1) + 1);
h = dsf*(roi(4) - roi(3) + 1);
Lc = w*h;

rowdex = zeros(Lc,1);
for i = 1:Lc
   j = rem(i-1,w*dsf)+1;
   k = rem(j-1,w)+1;
   rowdex(i) = w*dsf*fix((i-1)/(w*dsf)) + dsf*dsf*fix((k-1)/dsf) + dsf*fix((j-1)/w) + rem(k-1,dsf) + 1;
end
P = sparse(rowdex,1:Lc,1,Lc,Lc);
end

