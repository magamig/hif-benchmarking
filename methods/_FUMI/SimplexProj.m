function X = SimplexProj(Y)
%% This function prjected each row of Y into the canonical simplex
[N,D] = size(Y);
X = sort(Y,2,'descend');
% Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
Xtmp = (cumsum(X,2)-1).*repmat(1./(1:D),[N 1]);
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);


% [D,N] = size(Y);
% X = sort(Y,1,'descend');
% Xtmp = diag(sparse(1./(1:D)))*(cumsum(X,1)-1);
% X = max(bsxfun(@minus,Y,Xtmp(sub2ind([D,N],(1:N)',sum(X>Xtmp,1)))),0);

