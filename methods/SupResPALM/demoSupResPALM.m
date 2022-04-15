% -------------------------------------------------------------------------
% Hyperspectral Super-Resolution by Coupled Spectral Unmixing
% C. Lanaras, E. Baltsavias, K. Schindler
% ICCV 2015
% -------------------------------------------------------------------------
%
% DEMO
%
% Before running this script for the first time run compile.m to check all
% dependencies

addpath('include', 'reproject_simplex', 'sisal')

% Load an image from the CAVE database. Images must be in 2D format as
% described in the paper. To convert a 3D cube to 2D you can use
% hyperConvert2d.m
load('pompoms_ms.mat')

% Simulated Nikon D700 spectral response function (srf), the same as used
% in (Akhtar et al., 2014)
srf = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004 0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0
    0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003
    0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];

% Create the input data, spatial downsampling here is 32!
[ hyper, multi ] = hyperSynthetic( truth, srf, 32 );

% Define the number of material spectra
p=10;

%run PALM
tic
[E,A] = SupResPALM(hyper, multi, truth, srf, p);
toc

%Reconstruct the 3D cube
recIm3d = hyperConvert3d(E*A);

% Perform the evaluation
RMSE = hyperErrRMSE(truth,E*A)
SAM = hyperErrSam(truth,E*A)

