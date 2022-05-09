function image = unfold(Image3D,size,n)
k = size(1);
p = size(2);
l = size(3);
image = reshape(Image3D,[k*p,l]);
image = permute(image,[2,1]);
end