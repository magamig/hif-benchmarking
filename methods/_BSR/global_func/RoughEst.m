function [X_source]=RoughEst(XM,XH,XHd,psfY,nb_sub,P_dec)
[VXM,VXH,FBm,FBmC,FZ,FZ_s,FZC,FZC_s,~,ConvCBD,conv2im,conv2mat] = func_define(XM,XH,psfY,nb_sub);
XM_deg = func_blurringY(XM,psfY);
XM_deg = XM_deg(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
Cov_M  = covariance_matrix(XM_deg,XM_deg);
XHd_dec= var_dim(XHd,P_dec);
Cov_HM = covariance_matrix(XHd_dec,XM_deg);
%% Test the interpolation method
% X_mean = imresize(XHd_dec,[nr,nc],'bicubic');
% temp=Cov_HM/Cov_M*conv2mat(XM-imresize(XM_deg,[nr,nc],'bicubic'),size(XM,3));
X_mean=ima_interp_spline(XHd_dec,psfY.ds_r);
temp=Cov_HM/Cov_M*conv2mat(XM-ima_interp_spline(XM_deg,psfY.ds_r),size(XM,3));
X_source=X_mean+conv2im(temp,size(temp,1));