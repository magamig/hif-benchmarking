function [Image2D] = hyperConvert2D(Image3D)
if (ndims(Image3D) == 2)
    numBands = 1;
    [h, w] = size(Image3D);
else
    [h, w, numBands] = size(Image3D);
end
Image2D = reshape(Image3D, w*h, numBands).';
end
