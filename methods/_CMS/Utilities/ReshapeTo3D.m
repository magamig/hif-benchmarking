function Z = ReshapeTo3D(X , sz)
if length(sz)==2
    sz = [sz,1];
end
Z = zeros(sz);
for i = 1:sz(3)
    Z(:,:,i) = reshape(X(i,:),sz(1),sz(2));
end
