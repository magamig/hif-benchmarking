function Z = base_bicubic(X,sf)
sz = size(X);
w  = sqrt(sz(2));
Z = zeros(sz(1),sz(2)*sf*sf);
for i = 1 : sz(1)
    tX = X(i,:);
    tX = reshape(tX,w,w);
    tX2 = imresize(tX,sf);
%     tX2 = imresize(tX,sf,'nearest');
    Z(i,:) = tX2(:);
end