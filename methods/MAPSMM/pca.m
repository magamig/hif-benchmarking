function Out=pca(data)
% 
% INPUT
% data: (xdata*ydata)*band
% xdata
% ydata
%
% OUTPUT
% Out.pca: (xdata*ydata)*band
% Out.eig: band*1
% Out.a: band*band transformation matrix
mu=mean(data);
data=data-repmat(mu,size(data,1),1);
% Data covariance matrix
Vr=cov(data);

% Principle component analysis
[E,D]=eig(Vr);
[D,I]=sort(diag(D),'descend');
E=E(:,I');
Out.pca=data*E;
Out.a=E;
Out.eig=D;
Out.mean=mu;