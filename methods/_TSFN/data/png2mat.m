% Script for converting .png to .mat
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
Dir = 'CAVE';
Dir_out = 'truth';
Test_file       =    {'balloons_ms','beads_ms','cd_ms', 'chart_and_stuffed_toy_ms', 'clay_ms','cloth_ms', 'egyptian_statue_ms',...
                        'face_ms','fake_and_real_beers_ms','fake_and_real_food_ms','fake_and_real_lemon_slices_ms','fake_and_real_lemons_ms',...
                        'fake_and_real_peppers_ms','fake_and_real_strawberries_ms','fake_and_real_sushi_ms','fake_and_real_tomatoes_ms',...
                        'feathers_ms','flowers_ms','glass_tiles_ms','hairs_ms','jelly_beans_ms','oil_painting_ms','paints_ms',...
                        'photo_and_face_ms','pompoms_ms', 'real_and_fake_apples_ms', 'real_and_fake_peppers_ms','sponges_ms',...
                        'stuffed_toys_ms','superballs_ms','thread_spools_ms','watercolors_ms'
                        };

for i = 1:32
fpath         =   fullfile( fullfile(Dir, Test_file{i}), '*.png');
im_dir        =   dir(fpath);
bands         =   length(im_dir);
filestem      =   char(strcat(Dir, '/',Test_file{i}, '/', Test_file{i}));
for band = 1:bands
    filesffx    =  '.png';    
    if band < 10
        prefix  =  '_0';
    else
        prefix  =  '_';
    end    
    number      =   strcat( prefix, int2str(band) );
    filename    =   strcat( filestem, number, filesffx );
    Z           =   imread(filename);
    Z = Z(:,:,1);
    if  band==1
        sz        =   size(Z);
        s_Z       =   zeros(bands, sz(1)*sz(2));        
    end
    s_Z(band, :)   =   Z(:);
end
mv = max(s_Z(:));
truth    =  s_Z/mv;
truth = hyperConvert3D(truth, sz(1), sz(2));
save(fullfile(Dir_out, Test_file{i}),'truth');
end
