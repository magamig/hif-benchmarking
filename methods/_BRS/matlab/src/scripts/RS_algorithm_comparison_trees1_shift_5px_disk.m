%% algorithm comparison

% set path and default parameters
RS_init;

% set parameters for test case
dataset = 'trees1_shift_5px_disk'; % choose data set
groundtruth_available = true; % activate groundtruth comparison
lambda_u = [1e-1]; % image regularization parameter
lambda_k = [1e+1]; % kernel regularization parameter

folder_results = ['..', filesep, 'results', filesep, ...
    'algorithm_comparison'];
tracking = true;

% run iPALM
param_alg.algorithm = 'PALM'; %options: PALM, PAM
inertia = {0, 0.1, 0.2, 0.5};
param_alg.niter = 5000;

for i = 1 : length(inertia)
    param_alg.inertia = inertia{i};
    RS_run_example
end

% run PAM
param_alg.algorithm = 'PAM'; %options: PALM, PAM
inertia = {0};
param_alg.niter = 2000;
RS_run_example
