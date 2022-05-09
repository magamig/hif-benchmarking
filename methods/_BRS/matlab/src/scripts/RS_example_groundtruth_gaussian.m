%% ground truth example with gaussian kernel

% set path and default parameters
RS_init;

% set parameters for test case
dataset = 'trees1_shift_5px_gaussian'; % choose data set
groundtruth_available = true;
lambda_u = [1e-2, 1e-1, 1e-0]; % image regularization parameter
lambda_k = [1e+1]; % kernel regularization parameter

% run example
folder_results = ['..', filesep, 'results', filesep, 'example_', dataset];
RS_run_example
