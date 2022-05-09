%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  The implementation of Hybrid Monte Carlo method to evaluate the joint posterior distribution
%   AUTHOR: Qi WEI, University of Toulouse, qi.wei@n7.fr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear all;
setup;
Verbose='on';
generate=1;subMeth='PCA';FusMeth='Sparse';
scale=1;SNR_R=inf;seed=1;
%% Generate the data
[name_image,band_remove,band_set,nr,nc,N_band,nb_sub,X_real,XH,XHd,XHd_int,XM,VXH,VXM,psfY,psfZ_unk,...
    sigma2y_real,sigma2z_real,SNR_HS,SNR_MS,miu_x_real,s2_real,P_inc,P_dec,eig_val]=Para_Set(seed,scale,subMeth,SNR_R);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sparse fusion consists three parts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 1: Learn the rough estimation
learn_dic=1;train_support=1;Para_Prior_Initial;
X_source=RoughEst(XM,XH,XHd,psfY,nb_sub,P_dec);
%% Step 2: Learn the dictionary
[time_LD,Dic,supp]=Dic_Para(X_source,P_inc,learn_dic,train_support,X_real,0);
%% Step 3: Alternating optimization
[HSFusion.(FusMeth),Costime,diff_X,RMSE_sub,RMSE_org,tau_d_set,VXd_dec]=AlterOpti(X_source,XH,XM,psfY,...
    psfZ_unk,sigma2y_real,sigma2z_real,P_dec,P_inc,FusMeth,X_real,Dic,supp);
%% Evaluate the fusion results: Quantitative
[err_max.(FusMeth),err_l1.(FusMeth),err_l2.(FusMeth),SNR.(FusMeth),Q.(FusMeth),SAM_m.(FusMeth),RMSE_fusion.(FusMeth),...
    ERGAS.(FusMeth),DD.(FusMeth)] = metrics(X_real,HSFusion.(FusMeth),psfY.ds_r);
fprintf('%s Performance:\n SNR: %f\n RMSE: %f\n UIQI: %f\n SAM: %f\n ERGAS: %f\n DD: %f\n Time: %f\n',...
    FusMeth,SNR.(FusMeth),RMSE_fusion.(FusMeth),Q.(FusMeth),SAM_m.(FusMeth),ERGAS.(FusMeth),DD.(FusMeth),Costime.(FusMeth));
%% Display the fusion results: Qualitive
normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
temp_show=X_real(:,:,band_set);temp_show=normColor(temp_show);
figure(113);imshow(temp_show);title('Groundtruth')
temp_show=XHd_int(:,:,band_set);temp_show=normColor(temp_show);
figure(114);imshow(temp_show);title('HS image')
temp_show=mean(XM,3);temp_show=normColor(temp_show);
figure(115);imshow(temp_show);title('MS image')
temp_show=HSFusion.(FusMeth)(:,:,band_set);temp_show=normColor(temp_show);
figure(116);imshow(temp_show);title(['Fused image-' FusMeth])
name=[mat2str(clock) FusMeth '.mat'];save(name);