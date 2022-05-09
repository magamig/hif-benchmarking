%% urban example with 4 spectral channels

% set path and default parameters
RS_init;

% set parameters for test case
lambda_u = [1e-2]; % image regularization parameter
lambda_k = [1e+0]; % kernel regularization parameter

% select channels for reconstruction 
channels = 1 : 4;

for channel = channels;
    % choose data set
    dataset = sprintf('urban_ch%i_park', channel); 
    % run example
    folder_results = ['..', filesep, 'results', filesep, ...
        'example_urban_park_spectral_comparison'];
    RS_run_example

end

cd ..
addpath(genpath(pwd));
name = 'urban_park';
CreateColorImgs = true;
RS_rescale_and_generate_color_images;
