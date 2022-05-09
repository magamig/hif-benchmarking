%% comparison blind v nonblind image fusion

% set path and default parameters
RS_init;

% set parameters for test case
dataset = 'trees1_ch39_CNW'; % choose data set
lambda_k = [1]; % kernel regularization parameter

% run blind example
folder_results = ['..', filesep, 'results', filesep, ...
    'example_blind_v_nonblind', filesep, 'blind'];
lambda_u = [5e-1]; % image regularization parameter
RS_run_example

% run nonblind example
folder_results = [folder_results, filesep, '..', filesep, 'nonblind'];
param_alg.blind = false; % nonblind; don't update kernel
lambda_u = [1e-1]; % image regularization parameter
RS_run_example
