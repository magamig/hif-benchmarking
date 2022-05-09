function [dataHS,dataRGB] = load_hs_data(dataName,numWavelengths,suffix,rgbImageName)

scale = 1;

if nargin < 4, 
    dataRGB = double(imresize(imread([pwd '\data\' dataName '\' dataName(1:end-2) 'RGB.bmp']),scale));
else
    dataRGB = double(imresize(imread([pwd '\data\' dataName '\' rgbImageName]),scale));
end

[w,h,dontCare] = size(dataRGB);
dataHS = zeros(w,h,numWavelengths);

for i = 1:numWavelengths,   
    if i < 10, 
        dataHS(:,:,i) = double(imresize(imread(['data\' dataName '\' dataName '_0' num2str(i) suffix]),scale));
    else
        dataHS(:,:,i) = double(imresize(imread(['data\' dataName '\' dataName '_'  num2str(i) suffix]),scale));
    end    
end


