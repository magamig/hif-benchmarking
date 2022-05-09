%% comparison of TV with dTV

% set path and default parameters
RS_init;

% set parameters for test case
dataset = 'urban_ch2_park'; % choose data set
lambda_k = [1e+0]; % kernel regularization parameter

% select output folder
folder_results = ['..', filesep, 'results', filesep, 'example_TV_v_dTV'];

% run example dTV
lambda_u = [5e-2]; % image regularization parameter
RS_run_example

% run example TV
param_model.gamma = 0; % scalar factor of vectorfield: 0 <= gamma <= 1
lambda_u = [1e-3]; % image regularization parameter
RS_run_example
