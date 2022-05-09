clear;
close all;
clc;

rng(1);

data_name = "wadc"; 

deformation_type = "normal_nonrigid"; 

% "top_left, bot_center, center", 
% test the ability to learn PSF
psf_pos = "center";

% define the maximum deformation
alpha = 3; 

% load data
if data_name == "chikusei"
    load("./chikusei.mat");
    Scale_factor = 8;
    PSF_sigma = 1;
    all_srf = importdata('./Landsat.mat');  % B G R NIR Cirrus SWIR1 SWIR2
    srf = find_corr_srf(all_srf, wl); 
    SRF = srf(:,1:4);  % BGR NIR Cirrus SWIR1
    SRF = SRF ./ sum(SRF, 1);
elseif data_name == 'wadc'
    load("./wadc.mat");
    Scale_factor = 4;
    PSF_sigma = Scale_factor ./ 2.35482;
    all_srf = importdata('./Landsat.mat');
    srf = find_corr_srf(all_srf, wl); 
    SRF = srf(:,1:6);  % BGR
    SRF = SRF ./ sum(SRF, 1);
end

[nrows, ncols, nchannels] = size(img);
nrows = nrows ./ Scale_factor;
ncols = ncols ./ Scale_factor;

dir_name = data_name + "_" + deformation_type + "_" + num2str(alpha) + "_" + psf_pos;
mkdir(dir_name);

if alpha > 0
    if deformation_type == "normal_nonrigid"
        % bimodal Gaussian function    
        if data_name == "chikusei"
            num_lines = 4;
        elseif data_name == "wadc"
            num_lines = 5;
        end

        w_x = linspace(-3,3,floor(ncols));
        w_y = linspace(-3,3,floor(nrows/num_lines));
        [dx, dy] = meshgrid(w_x, w_y);
        W1 = exp(-((dx-1.5).^2+dy.^2)./ 1);
        W2 = exp(-((dx+1.5).^2+dy.^2)./ 1);
        W = W1+W2;
        W = W*alpha;

        fdx = zeros([nrows, ncols]);
        fdy = zeros([nrows, ncols]);

        if data_name == "chikusei"
            fdx = fdx + [-W;W;-W;W];
        elseif data_name == "wadc"
            fdx = fdx + [-W;W;-W;W;-W];
        end
        
        [y,x] = ndgrid(1:nrows,1:ncols);
        
        fig = figure(1);
        colormap gray; axis image; axis tight; hold on;
        quiver(x,y,fdx,fdy,0,'r');
        saveas(fig, dir_name+"/Displacement field.png");

    elseif deformation_type == "rigid"

        degree = 10;
        t = [10, 5];

        theta = degree * pi / 180;
        T = [cos(theta) -sin(theta) t(1);sin(theta) cos(theta) t(2);0 0 1];

    else
        error("请设置形变模式");
    end
end

HrHSI = img;
[LrHSI_ori, h] = downsamplePSF(HrHSI, PSF_sigma, Scale_factor, psf_pos);
HrMSI = downsampleSRF(img, SRF);

if alpha > 0
    LrHSI = gridsample(LrHSI_ori, x, y, fdx, fdy);
    [img_h, img_w, img_c] = size(LrHSI);
    LrHSI = LrHSI;
    save(dir_name+'/data.mat', 'HrHSI', 'LrHSI', 'HrMSI', 'x', 'y', 'fdx', 'fdy', 'SRF', 'Scale_factor', 'PSF_sigma', 'wl', 'h');
else
    LrHSI = LrHSI_ori;
    [img_h, img_w, img_c] = size(LrHSI);
    LrHSI = LrHSI;
    save(dir_name+"/data.mat", 'HrHSI', 'LrHSI', 'HrMSI', 'SRF', 'Scale_factor', 'PSF_sigma', 'wl', 'h');
end


if data_name == "chikusei"
    imshow_scalefactor = 3;
else
    imshow_scalefactor = 1;
end
    
fig = figure(3)
subplot(121); imagesc(retrieve_rgb(LrHSI_ori, wl).*imshow_scalefactor); axis image; title('Original');
axis off;
subplot(122); imagesc(retrieve_rgb(LrHSI, wl).*imshow_scalefactor); axis image; title('Elastic');
colormap gray
axis off;
saveas(fig, dir_name+"/Deformation Compare.png");

fig = figure(4)
imshowHRMSI = HrMSI(:,:,1:3);
imshow(imshowHRMSI(:,:,end:-1:1).*imshow_scalefactor);
axis off;
saveas(fig, dir_name+"/HrMSI.png")

fig = figure(5)
imshow(retrieve_rgb(LrHSI, wl).*imshow_scalefactor);
axis off;
saveas(fig, dir_name+"/LrHSI.png")


%%
function new_img = gridsample(img, x, y, fdx, fdy)
    for i = 1:size(img, 3)
        new_img(:,:,i) = griddata(x-fdx,y-fdy,double(img(:,:,i)),x,y);
    end
    new_img(isnan(new_img)) = 0;
end

%%

function new_img = downsampleSRF(img, SRF)
    SRF = SRF ./ sum(SRF, 1);
    [num_rows, num_cols, num_channels] = size(img);
    img = reshape(img, [num_rows*num_cols, num_channels]);
    new_img = img * SRF;
    new_n_channels = size(SRF, 2);
    new_img = reshape(new_img, [num_rows, num_cols, new_n_channels]);
end


%%

function srf = find_corr_srf(all_srf, wl)
    wl = wl' .* 1000;
    srf = [];
    for i=1:size(wl, 1)
        [~, corr_index] = min(abs(all_srf(:,1) - wl(i, 1)));
        srf = cat(1, srf, all_srf(corr_index, 2:end));
    end
end


%%

function [out_img, h] = downsamplePSF( img, sigma,stride, psf_pos )
    if psf_pos == "center"
        h = fspecial('gaussian',[stride,stride],sigma);
    elseif psf_pos == "top_left"
        h_init = fspecial('gaussian',[stride*2,stride*2],sigma);
        h = h_init(stride+1:end, stride+1:end);
    elseif psf_pos == "bot_center"
        h_init = fspecial('gaussian',[stride*2,stride*2],sigma);
        h = h_init(1:stride,stride/2:stride + stride/2-1);
    end
    
    [img_w, img_h, img_c] = size(img);
    for i = 1:img_c
        out_img(:,:,i) = downsample(downsample(conv2(img(:,:,i),h,'valid'),stride)',stride)';
    end
end

%%

function RGB = retrieve_rgb(data, wl, ideal_red_wl, ideal_green_wl, ideal_blue_wl, verbose)
%RETRIEVE_RGB Summary of this function goes here
%   Detailed explanation goes here
if nargin < 6
    verbose = 0;
end

if mean(wl) < 10 % the unit is micrometer
    wl = wl*1e3;
end

if nargin < 3
    ideal_blue_wl = 470;
    ideal_green_wl = 540;
    ideal_red_wl = 650;
end

options = struct('ideal_red_wl',ideal_red_wl,'ideal_green_wl', ...
    ideal_green_wl,'ideal_blue_wl',ideal_blue_wl);
rgb_ind = find_rgb_ind(wl,options);

red_ind = rgb_ind(1);
green_ind = rgb_ind(2);
blue_ind = rgb_ind(3);

blue_wl = wl(blue_ind);
green_wl = wl(green_ind);
red_wl = wl(red_ind);

if verbose
    disp(['Use ',num2str(blue_wl),'nm for blue, ', ...
        num2str(green_wl),'nm for green, ',num2str(red_wl),'nm for red.']);
end

I = data;
R = I(:,:,red_ind);
G = I(:,:,green_ind);
B = I(:,:,blue_ind);

RGB = cat(3,R,G,B);

RGB = RGB / max(RGB(:));
end

function rgb_ind = find_rgb_ind(wl,options)
if nargin < 2
    options = [];
end

ideal_red_wl = parse_param(options, 'ideal_red_wl', 650);
ideal_green_wl = parse_param(options, 'ideal_green_wl', 540);
ideal_blue_wl = parse_param(options, 'ideal_blue_wl', 470);

if mean(wl) < 10 % the unit is micrometer
    wl = wl*1e3;
end

[~,blue_ind] = min(abs(wl - ideal_blue_wl));
[~,green_ind] = min(abs(wl - ideal_green_wl));
[~,red_ind] = min(abs(wl - ideal_red_wl));

rgb_ind = [red_ind,green_ind,blue_ind];
end

function value = parse_param(options, field_name, default_value)
%PARSE_PARAM Summary of this function goes here
%   Detailed explanation goes here
if isempty(options) || ~isstruct(options) || ~isfield(options,field_name)
    value = default_value;
else
    value = options.(field_name);
end
end