function out = upsampling(data,w)

[rows,cols,bands] = size(data);
out = zeros(rows*w,cols*w,bands);
for b = 1:bands
    out(:,:,b) = imresize(reshape(data(:,:,b),rows,cols),w);
end