function [img] = hyperConvert2d(img)

if ~(ndims(img)==2 | ndims(img)==3)
    error('Input image must be 3D.')
end

[h,w,b] = size(img);

img = reshape(img,[w*h,b])';