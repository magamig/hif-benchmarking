function [X_BayesFusion]= BayesianFusion(XH,XM,psfZ,KerBlu,ratio,prior,start_pos)
%% This function is to implemnt the Bayesian Naive and Bayesian Sparse
%% representation based method in the following paper:

%[1] Qi Wei, Nicolas Dobigeon and Jean-Yves Tourneret, "Bayesian fusion of multi-band imagesn," 
%IEEE J. Sel. Topics Signal Process., vol. 9, no. 6, pp. 1-11, Sept. 2015. 
%[2] Qi Wei, Jos?Bioucas-Dias, Nicolas Dobigeon and Jean-Yves Tourneret, "Hyperspectral and 
%Multispectral Image Fusion based on a Sparse Representation," IEEE Trans. Geosci. and Remote 
%Sens., vol. 53, no. 7, pp. 3658-3668, July 2015. 
%[3] Qi Wei, Nicolas Dobigeon and Jean-Yves Tourneret, Fast Fusion of Multi-Band Images Based 
%on Solving a Sylvester Equationn", submitted.
% which is also mentioned in
%[4] Laetitia Loncan, Luis B. Almeida, Jose M. Bioucas-Dias, Xavier Briottet, 
%        Jocelyn Chanussot, Nicolas Dobigeon, Sophie Fabre, Wenzhi Liao, 
%        Giorgio A. Licciardi, Miguel Simoes, Jean-Yves Tourneret, 
%        Miguel A. Veganzones, Gemine Vivone, Qi Wei and Naoto Yokoya, 
%        "Introducing hyperspectral pansharpening," Geoscience and Remote Sensing
%        Magazine, 2015.

%If you use this code, please cite the above papers.
%% Input: 
%XH:        Hyperspectral Image
%XM:        Multispectral Image (or Panchromatic Image)
%overlap:   The overlapping bands of HS and MS image
%KerBlu:    The blurring kernel
%ratio:     The downsampling factor
%prior:     'Gaussian' or 'Sparse'
%start_pos  The starting point for the downsampling of HS
%% Output:
%X_BayesFusion:       The fused high-resolution HS image
%% -------------------------------------------------------------------------
%
% Copyright (March, 2015):        Qi WEI (qi.wei@n7.fr)
%
% BayesianFusion is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------
%l_pan=overlap(end);
%% Reshape the 3-D image to 2-D matrices
[n_dr,n_dc,N_HS]=size(XH);
[nr,nc,N_MS]=size(XM);

%% Shift the XM image
start_r=start_pos(1);
start_c=start_pos(2);
XM_shift=padarray(XM,[start_r-1 start_c-1],'circular','post');
XM_shift=XM_shift(start_r:end,start_c:end,:);

VXH=reshape(XH,[n_dr*n_dc N_HS])';
VXM=reshape(XM_shift,[nr*nc N_MS])';
%% Noise Variances
ChInv=eye(N_HS);
CmInv=eye(N_MS);
%% Subspace identification
subMeth='PCA';%subMeth='Hysime'
scale  =1;
[E_hyper, P_dec, ~, L]=idHSsub(XH,subMeth,scale);
sizeIM=[nr,nc,L];
%% Spectral mixture: spectral response R
%psfZ=[ones(size(overlap))/length(overlap) zeros(1,N_HS-l_pan)];
%figure; imagesc(psfZ);
%% Spatial Degradation: the blurring kernel B and HS Downsampling factor
% define convolution operator
psfY.B=KernelToMatrix(KerBlu,nr,nc);
psfY.ds_r=ratio; 
%% Interpolated the HS image with zeros
XH_int=zeros(nr,nc,N_HS);
XH_int(1:psfY.ds_r:end,1:psfY.ds_r:end,:)=reshape(VXH',[n_dr n_dc N_HS]);
VXH_int=reshape(XH_int,[nr*nc N_HS])'; 
%% The spline interpolated image as the prior mean of target image
Xd_dec = ima_interp_spline(var_dim(XH,P_dec),psfY.ds_r);
N_AO=1;
if strcmp(prior,'Sparse')
    learn_dic=1;train_support=1;
    if learn_dic==1
        [time_LD,D_s]=Dic_Para(XM_shift,E_hyper,learn_dic,train_support,0,inf);
        save Dic_Learn D_s time_LD -v7.3;
    else
        load('Dic_Learn.mat')
    end
    N_AO=10;
end
VX=SylvesterFusion(VXH_int,VXM,psfY,psfZ,ChInv,CmInv,E_hyper,Xd_dec,sizeIM,N_AO,prior);
%% Transforming from the subspace
VX_BayesFusion = E_hyper*VX;
X_BayesFusion  = reshape(VX_BayesFusion',[nr nc N_HS]);
%% Shift back
X_BayesFusion = padarray(X_BayesFusion,[start_r-1 start_c-1],'circular','pre');
X_BayesFusion = X_BayesFusion(1:nr,1:nc,:);