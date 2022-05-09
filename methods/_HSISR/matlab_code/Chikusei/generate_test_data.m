folder_path = "/Data/Chikusei/Test/";
factor = 0.125;
img_size = 512;
bands = 128;
save_dir = "/Data/Chikusei/test_dataset_x8/";

fileFolder=fullfile(folder_path);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
gt = zeros(numel(fileNames),img_size,img_size,bands);
ms = zeros(numel(fileNames),img_size*factor,img_size*factor,bands);
ms_bicubic = zeros(numel(fileNames),img_size,img_size,bands);

for i = 1:numel(fileNames)
    img_name = fileNames{i};
    load(fullfile(folder_path, img_name), 'test'); 
    gt = test;
    ms = single(imresize(gt, factor));
    ms_bicubic = single(imresize(ms,1/factor));
    gt = single(gt); 
    save_path = strcat(save_dir, img_name);
    save(save_path,'gt','ms','ms_bicubic');
end


