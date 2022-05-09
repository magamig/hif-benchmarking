clear all;clc;

path = './results';
folders = dir(path);
for i =3:length(folders)
    M = fullfile(path,folders(i).name);
    save_path = fullfile('./rgb_results/',folders(i).name(1:end-4))
    mkdir(save_path);
    load(M);
    for j = 1:31
        img = squeeze(res(1,j,:,:));
        imwrite(img,fullfile(save_path,strcat(folders(i).name(1:end-4),sprintf('_%.2d.png', j))));

    end
end
    