function [ mat3D ] = convert2Dto3Dpatchwise1( im2D, n )
[l N] = size(im2D);
x = sqrt(N);
y = sqrt(N);
indx = 1:n:x;
indxend = n:n:x;
indy = 1:n:y;
indyend = n:n:y;
im3D = zeros(x, y, l);
temp = im2D;
for j = 1:length(indy)
   for i = 1:length(indx)
       im = temp(:,1:n^2);
       patch3D = hyperConvert3D(im, n, n, l);
       temp(:, 1:n^2) = [];
       im3D(indx(i):indxend(i), indy(j):indyend(j), :) = patch3D;
   end
end
   mat3D = im3D; 
end
