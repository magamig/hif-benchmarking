function X = nls_coef(Y,D)
%--------------------------------------------------------------
% Estimate linear regression coefficients between hyperspectral and
% multispectral images via nonnegative least squares
%
% USAGE
%     X = nls_coef(Y,D)
% INPUT
%     Y : hyperspectral data (pixels, hs bands)
%     D : multispectral data (pixels, ms bands)
% OUTPUT
%     X : coefficients (ms bands, hs bands)
%
%--------------------------------------------------------------
p = size(Y,2);
m = size(D,2);
X = zeros(m,p);

H = D'*D;
lb = zeros(m,1);
opts = optimset('Algorithm','interior-point-convex','Display','off');
for i = 1:p
    y = Y(:,i);
    f = -(y'*D);
    x = quadprog(H,f,[],[],[],[],lb,[],[],opts);
    X(:,i) = x;
end