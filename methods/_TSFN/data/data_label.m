% Script for creating training/test data and generating HR-MSI
%
% Reference: 
% Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation
% Xiuheng Wang, Jie Chen, Qi Wei, C¨¦dric Richard
%
% 2019/05
% Implemented by
% Xiuheng Wang
% xiuheng.wang@mail.nwpu.edu.cn

clear;clc;
close all;

addpath('function')

sf    =    32;


foldertruth = 'truth';
foldertrain = 'train';
foldertest='test';


Test_file       =    {'balloons_ms','beads_ms','cd_ms', 'chart_and_stuffed_toy_ms', 'clay_ms','cloth_ms', 'egyptian_statue_ms',...
                        'face_ms','fake_and_real_beers_ms','fake_and_real_food_ms','fake_and_real_lemon_slices_ms','fake_and_real_lemons_ms',...
                        'fake_and_real_peppers_ms','fake_and_real_strawberries_ms','fake_and_real_sushi_ms','fake_and_real_tomatoes_ms',...
                        'feathers_ms','flowers_ms','glass_tiles_ms','hairs_ms','jelly_beans_ms','oil_painting_ms','paints_ms',...
                        'photo_and_face_ms','pompoms_ms', 'real_and_fake_apples_ms', 'real_and_fake_peppers_ms','sponges_ms',...
                        'stuffed_toys_ms','superballs_ms','thread_spools_ms','watercolors_ms'
                        };

for i = 1:32     
    
    i
    
S=load(fullfile(foldertruth, Test_file{i}), 'truth'); %load GT
S = S.truth;
[nr,nc,L] = size(S);
S_bar=Unfold(S,size(S),3);
R=create_F();

  %% genertate HR-MSI
MSI = hyperConvert3D((R*S_bar), nr, nc);
MSI1=Unfold(MSI,size(MSI),1);
MSI2=Unfold(MSI,size(MSI),2);
MSI3=Unfold(MSI,size(MSI),3);

% HR_HSI = imresize(HSI, sf, 'bicubic');

% img = cat(3, HR_HSI, MSI, S);
img = cat(3, S, MSI);
img = permute(img, [3,1,2]);
% disp(max(max(max(img))));
if i <= 20
    save(fullfile(foldertrain, strcat(num2str(i), '.mat')), 'img');
else
    save(fullfile(foldertest, strcat(num2str(i-20), '.mat')), 'img');
end
end