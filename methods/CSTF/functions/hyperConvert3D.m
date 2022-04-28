function [Image3D] = hyperConvert3D(Image2D, h, w)
[numBands, N] = size(Image2D);
if (1 == N)
    Image3D = reshape(Image2D, h, w);
else
   C= Image2D';
    Image3D = reshape(C, h, w, numBands); 
end
end

