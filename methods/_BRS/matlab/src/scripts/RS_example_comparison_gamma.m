%% gamma comparison on ground truth and real data

% set path and default parameters
RS_init;

% set parameters for GT test case
dataset = 'trees1_shift_5px_disk'; % choose data set
groundtruth_available = true;
gamma = [0.995, 0.995, 1]; % vectorfield constant
lambda_u = 1e-1;
lambda_k = 1e1;

% run example

folder_results = ['..', filesep, 'results', filesep, 'example_', dataset];

for gam = gamma
    param_model.gamma = gam;
    RS_run_example
end 

% set parameters for real data test case
dataset = 'trees2_ch108_NW'; % choose data set
groundtruth_available = false;
lambda_u = 1; % image regularization parameter
lambda_k = 1; % kernel regularization parameter

% run example
folder_results = ['..', filesep, 'results', filesep, 'example_', dataset];

for gam = gamma
    param_model.gamma = gam;
    RS_run_example
end 
