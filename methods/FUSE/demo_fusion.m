clear all
close all
clc

% This is a demo code to run Bayesian multi-band image fusion.
% See the main routine BayesianFusion.m for details
%
% Q. Wei, N. Dobigeon and J.-Y. Tourneret, 
% "Fast fusion of multi-band images based on solving a Sylvester equation", 
% IEEE Trans. Image Processing, vol. 24, no. 11, pp. 4109-4121, Nov. 2015.

addpath(genpath('func_global/'));
addpath(genpath('func_global/func_Sparse/'));

%% Loading image
load('data/moffet_ROI.mat');

I_REF = im; clear im;

%% Generating the HS and PAN image from the reference image
% ratio = size(I_PAN,1)/size(I_HS,1);
ratio = 5;
overlap = 1:41; % commun bands (or spectral domain) between I_PAN and I_HS
% overlap = 1:25; 
size_kernel=[9 9];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; % The starting point of downsampling
start_pos(2)=1; % The starting point of downsampling
[I_HS,KerBlu]=conv_downsample(I_REF,ratio,size_kernel,sig,start_pos);
I_PAN = mean(I_REF(:,:,overlap),3);

%% fusion

prior = 'Gaussian';
% prior = 'Sparse';

[X_BayesFusion]= BayesianFusion(I_HS,I_PAN,overlap,KerBlu,ratio,prior,start_pos);

%% plotting
figure
display_image(I_PAN)

figure
display_image(I_HS(:,:,[28 19 11]))

figure
display_image(X_BayesFusion(:,:,[28 19 11]))
