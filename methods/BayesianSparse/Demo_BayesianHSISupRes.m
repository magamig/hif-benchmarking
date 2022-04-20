% Please cite the following work, if you use the code:
% N. Akhtar, F. Shafait and A. Mian, "Bayesian Sparse Representation for
% Hyperspectral image super-resolution", in CVPR 2015.
% For any issues with the code, please contact
% naveed.akhtar@research.uwa.edu.au

% Demo for hyperspectral super resolution using non-parametric Bayesian
% sparse representation.
% The demo implements the approach sequentially, therefore the default
% values of the total iterations are set to smaller values. However, the results remain comparable due to the fast
% convergence of the approach.


%------------------------------------------------------------------
% Possible choices
%------------------------------------------------------------------
% (a)Directly run the demo:
%       It will read the image provided inside the demo folder
% (b)Use your own image:
%       Go through the following steps:
%       - Save the image in the current folder as a matlab structre Name.im
%       (Name.im should return the M x N x L hyperspectral image )
%       - Set param.HSI = 'Name'
%       - Run the script

%------------------------------------------------------------------
% Parameter settings 
%------------------------------------------------------------------
param.HSI = 'img2';        % Image to be tested
param.a0 = 1e-6;               
param.b0 = 1e-6;               
param.c0 = 1e-6;
param.d0 = 1e-6;
param.e0 = 1e-6;
param.f0 = 1e-6;
param.DLiterations = 50000;  % Total iterations for the dictionay learning stage. Last 100 are used for averaging
param.SCiterations = 100;   % Number of iterations per single run of the sparse coding stage
param.Q = 32;               % Parameter Q: total number of times, sparse coding is performed independently. 32 is generally good for harvard database.

%------------------------------------------------------------------
% Run code
%------------------------------------------------------------------
BayesianHSISupRes(param)


