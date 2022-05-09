
close all,clear all
load('PatchOutput-cave.mat');
sz = size(output,2)*8;
num=size(output,1)/8/8;
c=size(output,4);
X=zeros(sz,sz,c);
O=zeros(num,sz,sz,c);
for i =1:num
    for H = 1:8
        for W = 1:8
            ind = (i-1)*64+(H-1)*8+W;
            img = squeeze(output(ind,:,:,:));
        X((H-1)*64+1:H*64,(W-1)*64+1:W*64,:) = img;
        imshow(img(:,:,[30,15,2]),[]);
        end
    end
%     close all
    figure,imshow(X(:,:,[31,24,2]),[]);    
    O(i,:,:,:) = X;
end
output=O;
save('Output-cave.mat','output');

