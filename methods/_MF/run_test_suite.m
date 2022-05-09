% run_test_suite.m
clear;
clc;

addpath('factorization code');
addpath('data');
addpath('helpers');
addpath('hyperspectral specific code');

fileNames   = {'balloons_ms'}; %,'beads_ms','cd_ms','fake_and_real_peppers_ms','flowers_ms','oil_painting_ms','photo_and_face_ms','sponges_ms','thread_spools_ms'};
WAVELENGTHS = 31;
SUFFIX      = '.png';
P_rgb = load_nikon_rgb(400,10,31); 

rmseVals = nan(length(fileNames),1);

for i = 1:length(fileNames),
    
    disp(fileNames{i});    
    [dataHS_hr,dataRGB_hr] = load_hs_data(fileNames{i},WAVELENGTHS,SUFFIX);
    
    % Discard dataRGB_hr (real file taken with unknown camera).
    % Generate synthetic RGB image from our input data
    dataRGB_hr = generate_fake_rgb(dataHS_hr,P_rgb);
    
    % Generate synthetic low-res measurements
    dataHS_lr = generate_low_spatial_res_measurement(dataHS_hr);
    
    % Test reconstruction
    [recHS,facBasis,facCoeff] = unmix_and_reconstruct(dataHS_lr,dataRGB_hr,P_rgb);

    % Evaluate RMSE
    rmse = eval_rmse(recHS,dataHS_hr,2^-8);
    
    disp(['Finished! RMSE: ' num2str(rmse)]);
    
    rmseVals(i) = rmse;     
    mkdir(['results\' fileNames{i}]);
    save_hs_data(['results\' fileNames{i}],recHS);
    
    save(['results\' fileNames{i} '\fac.mat'],'recHS','facBasis','facCoeff');
    
    pause(.1);
end