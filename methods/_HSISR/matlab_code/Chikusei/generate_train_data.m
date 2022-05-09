path = '/Data/Chikusei/Train/';
patch_size = 128;
stride = 64;
factor = 0.125;
save_dir = "/Data/Chikusei/train_dataset_x8/";

file_folder=fullfile(path);
file_list=dir(fullfile(file_folder,'*.mat'));
file_names={file_list.name};


% store cropped images in folders
for i = 1:1:numel(file_names)
    name = file_names{i};
    name = name(1:end-4);
    load(strcat(path,file_names{i}));
    crop_image(img, patch_size, stride, factor, name, save_dir);
end