function save_hs_data(dataName,dataHS)

for i = 1:size(dataHS,3),   
        imwrite(dataHS(:,:,i)/2^16,[ dataName '\' num2str(i) '.png'],'BitDepth',16);
end