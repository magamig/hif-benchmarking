file_folder = fullfile('Data/CAVE/CAVE_raw/');
file_list = dir(fullfile(file_folder, '*_ms'));
file_names = {file_list.name};
file_names = file_names(21:32);
file_dir = '/Data/CAVE/factor_x4/test/';
factor = 0.25;

for i = 1:1:numel(file_names)
    name = file_names{i};
    if (i ~= 12)
       img_folder = fullfile(file_folder, name, name);
    else
       subname = 'watercolors_ms_keke';
       img_folder = fullfile(file_folder, name, subname);
    end
    %img_folder = fullfile(file_folder, name, name);
    img_list = dir(fullfile(img_folder, '*.png'));
    img_names = {img_list.name};
    gt_img = zeros(512, 512, 31);
    for j = 1:1:numel(img_names)
        img_name = img_names{j};
        img = imread(fullfile(img_folder, img_name));
        gt_img(:,:,j) = img;
    end
    gt_img = double(gt_img)./65535;
    ms = single(imresize(gt_img, factor));
    ms_bicubic = single(imresize(ms,1/factor));
    gt = single(gt_img);
    file_path = strcat(file_dir, name, '.mat');
    save(file_path,'gt','ms','ms_bicubic','-v6');   
    
end
   
   