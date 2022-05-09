%% tree example

% set path and default parameters
RS_init;

% set parameters for test case
dataset = 'trees2_ch108_NW'; % choose data set
lambda_u = [1e-1, 1e-0, 1e+1]; % image regularization parameter
lambda_k = [0, 1e+0, 1e+2]; % kernel regularization parameter

% run example
folder_results = ['..', filesep, 'results', filesep, 'example_', dataset];
RS_run_example
