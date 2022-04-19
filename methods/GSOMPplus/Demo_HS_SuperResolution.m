%Please cite the following paper, if you use the code:
% Akhtar, Naveed, Faisal Shafait, and Ajmal Mian. "Sparse Spatio-spectral Representation for Hyperspectral Image Super-Resolution." Computer Vision–ECCV 2014. Springer International Publishing, 2014. 63-78.

%Please contact naveed.akhtar@research.uwa.edu.au for any issues regarding
%the code.

% Running the file will run the demo of the presented approach on the
% 'Faces' image. 
% The demo takes around 3 minutes on Intel Core i7-2600 CPU at 3.4 (8 GB
% RAM)
%
%-------------------------------------------
% Possible choices 
%-------------------------------------------
% (a)Simply run the demo without SPAMS***(see below) installed. (set
% param.spams = 0)
%       It will read the image in the folder and use the already learned
%       dictionary to generate the results.
% (b)Run the demo with SPAMS already installed. (set param.spams = 1)
%       It will run the complete approach, including the dictionary learning step. 
% (c)Use your own image. (SPAMS must be installed)
%       Go through the following steps to use your own image
%       - Save the image in the current folder as a matlab structre Name.im
%       (Name.im should return the M x N x L hyperspectral image)
%       - Set param.spams = 1
%       - Set param.HSI = 'Name'
%       - Run the script

%---------------------------------------------
% Setting the parameters to the default values
%-----------------------------------------------
param.spams = 0;        % Set = 1 if SPAMS***(see below)is installed, 0 otherwise
param.L = 20;           % Atoms selected in each iteration of G-SOMP+
param.gamma = 0.99;     % Residual decay parameter
param.k = 300;          % Number of dictionary atoms
param.eta = 10e-9;      % Modeling error
param.HSI = 'Faces';    % Image to be tested

%-----------------------------------------
% Run code
%------------------------------------------
superResolution(param)
%***SPAMS (SPArse Modeling Software) is an open source tool for sparse modeling.
%It can be easily searched on the internet and installed. 
%We use SPMAS to implement the online dictionary learning approach [30].