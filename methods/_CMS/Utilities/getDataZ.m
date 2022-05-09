function dataZ = getDataZ()
% This function aims to get high spatial resolution HSI from CAVE
% datasets, combine 31 images on each bands together 

dataZ = [];
for i = 1 : 31
    picNum = sprintf('%02d',i);
    dataPathT =['.\data\watercolors_ms\watercolors_ms_',picNum,'.png'];
    dataT = imread(dataPathT);
%     dataT = uint8(sum(dataT,3)/3);    %只针对watercolors_ms数据集
    dataZ = cat(3,dataZ,dataT);
end
