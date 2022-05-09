file_folder = fullfile('/Data/CAVE/CAVE_raw/');

patch_size = 64;
stride = 32;
scaling_factor = 0.25;
save_dir = "/Data/CAVE/factor_x4/";


file_list = dir(fullfile(file_folder, '*_ms'));
file_names = {file_list.name};
file_names = file_names(1:20);


for i = 1:1:numel(file_names)
    name = file_names{i};
    img_folder = fullfile(file_folder, name);
    img_list = dir(fullfile(img_folder, name, '*.png'));
    img_names = {img_list.name};
    gt_img = zeros(512, 512, 31);
    for j = 1:1:numel(img_names)
        img_name = img_names{j};
        img = imread(fullfile(img_folder, name, img_name));
        gt_img(:,:,j) = img;
    end
    crop_image(gt_img, patch_size, stride, scaling_factor, name, save_dir);
end
   
   