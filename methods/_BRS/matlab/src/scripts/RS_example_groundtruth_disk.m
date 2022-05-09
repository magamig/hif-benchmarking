%% ground truth example with disk kernel

% set path and default parameters
RS_init;

% set parameters for test case
dataset = 'trees1_shift_5px_disk'; % choose data set
groundtruth_available = true;
lambda_u = [1e-1]; % image regularization parameter
lambda_k = [0, 1e+1, 1e+2]; % kernel regularization parameter

% run example
folder_results = ['..', filesep, 'results', filesep, 'example_', dataset];
RS_run_example
