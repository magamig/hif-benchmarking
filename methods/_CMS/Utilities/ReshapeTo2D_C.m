function Z = ReshapeTo2D_C(X)
sz = size(X);
if length(sz)==2
    sz = [sz,1];
end
Z = zeros(sz(1)*sz(2),sz(3));
for i = 1 : sz(3)
    t = X(:,:,i);
    Z(:,i) = t(:);
end