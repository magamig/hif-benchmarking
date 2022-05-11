function [Image3D] = hyperConvert3D(Image2D, h, w, numBands)
[numBands, N] = size(Image2D);
if (1 == N)
    Image3D = reshape(Image2D, h, w);
else
    Image3D = reshape(Image2D.', h, w, numBands); 
end
end

