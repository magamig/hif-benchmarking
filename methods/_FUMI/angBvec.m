function z=angBvec(X,Y)
%        y=angBvec(X,Y)
%
% compute angles between vectors X and Y in degress

%normalize X

X = X./repmat(sqrt(sum(X.^2)),size(X,1),1);
Y = Y./repmat(sqrt(sum(Y.^2)),size(Y,1),1);
z= acos(sum(X.*Y)-eps)*180/pi;



