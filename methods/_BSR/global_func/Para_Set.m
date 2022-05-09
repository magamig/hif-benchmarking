function [name_image,band_remove,band_set,nr,nc,N_band,nb_sub,X_real,XH,XHd,XHd_int,XM,VXH,VXM,...
    psfY,psfZ_unk,s2y_real,s2z_real,SNR_HS,SNR_MS,miu_x_real,s2_real,P_inc,P_dec,eig_val]=Para_Set(seed,scale,subMeth,SNR_R)
%% Set the size of reference image
% rng(seed);
% randn('state','v5normal');
% global miu_x_real;global s2_real;global sigma2y_real;global sigma2z_real;
% nr=32;nc=32;
% N_start=100;nb_sub=5; % The starting position and the dimension of subspace
%% Select the image source
% name_image = 'moffet_ROI3_bis.mat';N_band=172;band_set=[26 17 10];nb=4; N_start_r=50;N_start_c=50;nb_sub=10;
% nr=128;nc=128;name_image = 'moffet_ROI3_bis.mat';N_band=176;band_set=[20 11 4];N_start_r=0; N_start_c=0;nb_sub=10;%For MS+HS 
% nr=64;nc=64;name_image = 'moffet_ROI3_bis.mat';N_band=3;N_start=0;nb_sub=3;band_set=[]; %For Pan+MS 
% nr=64;nc=64;name_image = 'moffet_ROI3_bis.mat';N_band=177;band_set=[20 11 4];N_start=0;nb_sub=4; %For Pan+HS 
nr=128;nc=128;name_image='pavia.mat';N_band=103-10;band_set=[45 25 8];N_start_r=50;N_start_c=50;nb_sub=5;  %For MS+HS 
% nr=512;nc=256;name_image='pavia.mat';N_band=103-10;band_set=[45 25 8];N_start_r=90;N_start_c=00;nb_sub=5;  %For MS+HS 
% nr=64;nc=64;name_image='pavia.mat';N_band=50;band_set=[45 25 8];N_start=50;nb_sub=5;%For Pan+HS 
% name_image='pleiades_subset.mat';N_band=103;
% SNR_R=SNR_R_set(i_R);
% SNR_R=inf;
%% Constructing the groundtruth image
[X_real,band_remove]= real_image(name_image,nr,nc,N_band,N_start_r,N_start_c);%X_temp=X_real;
% VX_real=reshape(X_real,[nr*nc N_band])';nb_sub=8;
%%geo_method='vca';[endm P_dec P_inc Y_bar endm_proj VX_real_proj] = find_endm(VX_real,nb_sub+1,geo_method);
% [endm, indice, VX_real] = vca(VX_real,'Endmembers',nb_sub,'verbose','off');
% X_real=reshape(VX_real',[nr nc N_band]);
%%figure;imagesc(X_real(:,:,band_set));
%% Processing the groundtruth to make sure it really lives in the subspace
% [w,Rn] = estNoise(VX_real);
% [Nbs_real,P,~]=hysime(VX_real,w*0,Rn*0);
% P=P(:,1:10);
% X_real=var_dim(var_dim(X_real,P'),P);
%% Set the noise power of HS and MS data
SNR_HS=[35*ones(N_band-50,1);30*ones(50,1)];
% SNR_HS=[30*ones(N_band-5,1);30*ones(5,1)];
% SNR_HS=30*ones(N_band,1)+randn(N_band,1)*0;
% SNR_HS=30*ones(N_band,1);
%SNR_MS=[30*ones(nb-3,1);30*ones(3,1)];
SNR_MS=30;
%% Generate the HS and MS images
[psfY,psfZ_unk,XH,XM,s2y_real,s2z_real]= HS_MS(X_real,SNR_HS,SNR_MS,SNR_R,name_image,band_remove);
%% Abstract the mean
% XH_mean=repmat(mean(mean(XH,1),2),[size(XH,1) size(XH,2) 1]);XH=XH-XH_mean;
% XM_mean=repmat(mean(mean(XM,1),2),[size(XM,1) size(XM,2) 1]);XM=XM-XM_mean;
%% Downsampled HS image
XHd=XH(1:psfY.ds_r:end,1:psfY.ds_r:end,:);   
XHd_int=ima_interp_spline(XHd,psfY.ds_r);
%% HS subspace identification: Identifying the subspace where HS data live in
temp=reshape(XHd,[size(XHd,1)*size(XHd,2) N_band])';
% temp=temp-repmat(mean(temp,2),[1,size(temp,2)]); % or equally temp=bsxfun(@minus, temp, mean(temp,2));
if strcmp(subMeth,'Hysime')
    [w,Rn] = estNoise(temp);
    [nb_sub,P_vec,eig_val]=hysime(temp,w,Rn);%diag(sigma2y_real)
elseif strcmp(subMeth,'PCA')
    [P_vec,eig_val]=fac(XHd);
%     [nb_sub,~,~]=hysime(temp,w,Rn);%diag(sigma2y_real)
    PCA_ratio=sum(eig_val(1:nb_sub))/sum(eig_val); %[P_vec,eig_val]=fac(X_real);
    P_vec=P_vec(:,1:nb_sub); % Each column of P_vec is a eigenvector
end
if scale==1
    P_dec=diag(1./sqrt(eig_val(1:nb_sub)))*P_vec';
    P_inc=P_vec*diag(sqrt(eig_val(1:nb_sub)));
elseif scale==0
    P_dec=P_vec';
    P_inc=P_vec;
end
% tempX=reshape(X_real,size(X_real,1)*size(X_real,2),size(X_real,3))';
% disp([mean2((Ek*Ek'*tempX-tempX).^2) mean2((P_inc*P_dec*tempX-tempX).^2)]);

% if size(P_dec,1)==size(P_dec,2)
%     P_dec=eye(size(P_dec));
% end
%% Project the image to subspace and project it back
miu_x_real=squeeze(mean(mean(X_real,1),2)); 
s2_real = cov(reshape(X_real,[size(X_real,1)*size(X_real,2) size(X_real,3)]));

VXH=reshape(XHd,size(XHd,1)*size(XHd,2),size(XHd,3))';
VXM=reshape(XM,size(XM,1)*size(XM,2),size(XM,3))';
% Rotate the subspace transform matrix
% [P_vec,Q_rot]=rotate_subspace(P_vec,N_band);
% X_real=var_dim(var_dim(X_real,P_vec','dec'),P_vec','inc');