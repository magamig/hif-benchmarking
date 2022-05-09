clear;close all;
%% settings
foldertest = '超分数据/初始化/';
folderlabel='超分数据/残差/';
size_input=32;
size_label=32;
stride =16;

savepath = 'train_res_noise.h5';

m=20;%使用训练的图片的数量

%% initialization
data = zeros(size_input, size_input, 31, 1);
label = zeros(size_label, size_label, 31, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate test data
for i=1:m
    
    i
    
    img=load(strcat(foldertest,num2str(i),'.','mat'));
    imgl=load(strcat(folderlabel,num2str(i),'.','mat'));
    
    image_test = img.b;
%     image_test(:,:,4:34) = image_test(:,:,4:34)/100;
    image_label = imgl.b;

%     imwrite(image_input, strcat('train_label/processing',num2str(i), '.bmp'));
    
    image_input=image_test;
    image_output=image_label;
    
    [hei,wid,c]=size(image_input);

    for x = 1 : stride : hei-size_input+1%这里sdtrde为步长
        for y = 1 :stride : wid-size_input+1
            
            subim_input = image_input(x : x+size_input-1, y : y+size_input-1,:);
            subim_label = image_output(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);

            count=count+1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
        end
    end
end
%% processing data for matlab
order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);


%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
