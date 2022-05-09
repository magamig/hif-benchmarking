%% 1.The prior of HS noise variance
PowH=squeeze(mean(mean(XHd.^2,1),2));
VarH=PowH.*(10.^(-(SNR_HS)/10)); 
Para_Prior.nuH=size(XHd,1)*size(XHd,2)*1e100+3; 
Para_Prior.gammaH=(Para_Prior.nuH+2)*VarH;
%% 2.The prior of MS noise variance
PowM=squeeze(mean(mean(XM.^2,1),2)); % The power of MS image
VarM=PowM.*(10.^(-(SNR_MS)/10));     % The rough estimation of noise power
Para_Prior.nuM=size(XM,1)*size(XM,2)*1e100+3;        
Para_Prior.gammaM=(Para_Prior.nuM+2)*VarM; % The informative prior for noise variances
%% 3.The Prior Mean 
%% Way 1: set miu_x as the interpolated HS image
Para_Prior.miu_x=reshape(var_dim(XHd_int,P_dec),[nr*nc nb_sub]);%% Reshape miu_x if miu_x is tbe interpolated HS image
%% 4.Prior for the covariance matrix
%     tempXH=func_blurringY(XHd_int,psfY);
%     Cov_U=covariance_matrix(var_dim(XHd,P_dec),var_dim(tempXH(1:psfY.ds_r:end,1:psfY.ds_r:end,:),P_dec));
%     Cov_U=tempX*tempX'/size(tempX,2);
%     Phi=(eta+nb_sub+1)*((tempX*tempX'/size(tempX,2)));
%     tempX=var_dim(X_real-XHd_int,P_dec);
%     tempX= var_dim(X_real,P_dec)-conv2im(VX_dec,nb_sub);
%     tempX=X_source-var_dim(XHd_int,P_dec);
%     tempX=conv2mat(tempX,nb_sub);

tempXH=func_blurringY(XHd_int,psfY);
% % Cov_U0=covariance_matrix(var_dim(XHd,P_dec),var_dim(tempXH(1:psfY.ds_r:end,1:psfY.ds_r:end,:),P_dec));
temp=var_dim(XHd,P_dec)-var_dim(tempXH(1:psfY.ds_r:end,1:psfY.ds_r:end,:),P_dec);
Cov_U0=covariance_matrix(temp,temp);
% tempX=reshape(var_dim(X_real,P_dec),[nr*nc nb_sub])-Para_Prior.miu_x;
% Cov_U0=tempX'*tempX/size(tempX,1);
if strcmp(name_image,'pavia.mat')  
    Cov_U0=eye(nb_sub);
end
Para_Prior.eta=(nb_sub+3)+1e0*nr*nc; %% Control the informativeness
% Cov_U0=eye(nb_sub)*var(XHd(:)); % The mean of Prior
Para_Prior.Phi=(Para_Prior.eta+nb_sub+1)*Cov_U0;
%% Initialization
Para_Prior.invCov_U0=inv(Cov_U0);

% if strcmp(prior,'Uniform') 
% %     s2_inv=eye(L);
% %     miu_x=zeros(size(X,3),1);
%     s2_inv=[];
%     miu_x=[];  
% elseif strcmp(prior,'GMM')
%     s2_inv=0.01*repmat(eye(L),[1 para.Num_G]);
%     if strcmp(mean_prior,'empirically')
%         %% Way 1: set miu_x as the interpolated HS image
%         miu_x=var_dim(Y(1:psfY.ds_r:end,1:psfY.ds_r:end,:),P_dec);
%         miu_x=ima_interp_spline(miu_x,psfY.ds_r);
%         miu_x=reshape(miu_x,[N_row*N_col L]);%% Reshape miu_x if miu_x is tbe interpolated HS image
%     else
%         %% Way 2: sample miu_x for all pixels
%         miu_x=zeros(para.Num_G,L);    
%     end
%     %% Prior for hyperparameters
%     Y_tem=Y(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
%     %% The prior mean for hyperparameter mean
%     miu_x0=P_dec*mean(reshape(Y_tem,[size(Y_tem,1)*size(Y_tem,2) size(Y_tem,3)]))';
% %     sigma2_x0=100;
%     %% Way 1: Assign the prior of hyperparameter empirically 
% %     s2_0=P_dec*cov(reshape(Y_tem,[size(Y_tem,1)*size(Y_tem,2) size(Y_tem,3)]))*P_dec';
% %     s2inv_0=inv((s2_0+s2_0')/2);
% %     eta=nb_sub+1+2+0e-3*N_row*N_col; % This is to enforce the informative prior
% %     s2inv_0=s2inv_0/(eta-nb_sub-1);    
%      %% Way 2: Assign the prior of hyperparameter non-informatively
%      nb_sub=size(P_dec,1);
%      s2inv_0=1e-0*eye(nb_sub);
%      eta=nb_sub+3+1e-2*N_row*N_col;
%      
% elseif strcmp(prior,'Dic')
%     s2_inv=0.01;
%     miu_x=var_dim(Y,P_vec,'dec');
% end