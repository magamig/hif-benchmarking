function PSNR = get_PSNR(C_IMGS,maxval)
% Get the PSNR Measure
% Inputs:
%   C_IMGS
% Outputs:
%   PSNR.band
%   PSNR.avg
r = size(C_IMGS,1); w = size(C_IMGS,2)/3; bands = size(C_IMGS,3);
MSE = zeros(bands,1);
PSNR.band = zeros(bands,1);
PSNR.avg = 0;
for band = 1:bands
   for row = 1:r
       for col = 2*w+1:3*w
           MSE(band) = MSE(band) + C_IMGS(row,col,band)^2;
       end
   end
   MSE(band) = MSE(band)/(r*w);
   %PSNRt = max(max(C_IMGS(1:r,w+1:2*w,band)))^2/MSE(band);
   PSNR.band(band) = 10*log10((maxval^2)/MSE(band));
   %PSNR.avg = PSNR.avg + PSNRt;
end
%PSNR.avg = 10*log10(PSNR.avg/bands);
PSNR.avg = 10*log10((maxval^2)/mean(MSE));
end