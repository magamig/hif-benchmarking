function [ mat2D ] = convert3Dto2Dpatchwise( im3D, n )
[y, x, d] = size(im3D);
np = floor(x/n); 
nq=floor(y/n);
s = 1:n:x;
e = n:n:x;
temp = zeros(y,n,d);    
mat2D = [];
for j = 1:np           
    temp = im3D(:,s(j):e(j),:);   
    temp2 = zeros(n,n,d);
    for k = 1:nq
        temp2 = temp(s(k):e(k),:,:);  
        patch2D = reshape(temp2, n*n, d).';
        mat2D = [mat2D patch2D];
    end   
end
end