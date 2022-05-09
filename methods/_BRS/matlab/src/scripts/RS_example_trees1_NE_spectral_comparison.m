%% Tree example with many spectral channels

% set path and default parameters
RS_init;

% set parameters for test case
lambda_u = [1e+0]; % image regularization parameter
lambda_k = [1e+0]; % kernel regularization parameter

% select channels for reconstruction 
channels = 12 : 18 : 102;

for channel = channels

    % choose data set
    dataset = sprintf('trees1_ch%i_NE', channel);

    % run example
    folder_results = ['..', filesep, 'results', filesep, ...
        'example_trees1_NE_spectral_comparison'];
    RS_run_example
    
end

cd ..
addpath(genpath(pwd));
name = 'trees1_NE';
CreateColorImgs = false;
RS_rescale_and_generate_color_images;
