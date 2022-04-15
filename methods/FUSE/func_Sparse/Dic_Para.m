%% Learn the dictionary
function [time_LD,varargout]=Dic_Para(X_source,P_inc,learn_dic,train_support,X_real,Mem_lim)
% Para_Prior_Initial;
nb_sub=size(P_inc,2);
% [VXM,VXH,FBm,FBmC,FZ,FZ_s,FZC,FZC_s,~,ConvCBD,conv2im,conv2mat] = func_define(XM,XH,psfY,nb_sub);
method='rotate';
% learn_dic = 0; train_support= 0; % learn the dictionary ?
% method='Pan';
% if strcmp(method,'rotate1')
%     [P_dec,Q_rot]=rotate_subspace(P_dec,N_band);
%     % Judge which Dic is to be used
%     Dic_index=zeros(size(P_dec,1),1);
%     for k=1:size(P_dec,1)
%         [~,Dic_index(k)]=min(sum((repmat(P_dec(k,:),size(psfZ,1),1)-psfZ).^2,2));
%     end
% end
%% Assign the learned ditionary for each band
% 1st way to learn temp
% temp =(repmat(max(psfZ),size(psfZ,1),1)==psfZ);  
% temp(:,sum(temp)==size(temp,1))=0; % Make the spectral response 2-value:0 or 1
% [xl,band_coverd]=find(temp);  % find the position of covered band
% band_uncovered=setdiff(1:size(psfZ,2),band_coverd);  % find the poisition of unconverd band by MS image
% % band_u_new=zeros(size(band_uncovered)); pos=zeros(1,length(band_uncovered));
% for i=1:length(band_uncovered)
%     [~,pos]=(min(abs(band_uncovered(i)-band_coverd))); % find the covered band which is closest to the uncovered band
% %     band_u_new(i)=band_coverd(pos(i)); 
%     temp(xl(pos),band_uncovered(i))=1;  % assign the corresponding MS band
% end
%2nd way to calculate temp:
% temp=(psfZ'*psfZ+1e-10*eye(size(psfZ,2)))\psfZ';temp=temp';temp(temp<0)=0;
% temp=temp./repmat(sum(temp),size(temp,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Parameters for Processing method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
%% Strategies to select the source image to train the dictionary and support
% Method 1: Pan, the weighting sum of the MS bands
% X_source=mean(XM,3);

% Method 2: interpolation of HS data
% X_source=var_dim(imresize(XHd,[nr,nc],'bicubic'),P_dec','dec');
% Method 3: Normalized MS data
%     X_source=zeros(size(XM));
%     for i=1:size(XM,3)
%         norm_p=sqrt(mean2(XM(:,:,i).^2));
%         X_source(:,:,i)=XM(:,:,i)/norm_p;
%     end

% Method 4: MS data
%     X_source=XM;

% Method 5: Results of Wavelet fusion
% load('HSFusion.mat');X_source=var_dim(imresize(HSFusion.Wavelet,[nr,nc],'bicubic'),P_dec','dec');   

% Method 6: Normalized MS data
%     X_source=func_blurringZ(X_real,psfZ_unk);
%     temp=repmat(mean(XM,3),[1 1 size(P_dec,1)]);
%     ratio=0.75;[P_vec_s,~]=fac(XM,ratio);P_vec_s=P_vec_s(:,1);
%     temp=var_dim(XM,P_vec_s,'dec');
%     temp=repmat(temp,[1 1 size(P_dec,1)]);

% Method 7: Restored HS data from MS data
%      X_source_tem=(psfZ_s'*psfZ_s+1e-4*eye(size(P_dec,1)))\psfZ_s'*conv2mat(XM,nb);
%      X_source_tem=RV'*conv2mat(XM,nb);
%      X_source=conv2im(X_source_tem,size(P_dec,1));

% Method 8: Reference image    
% X_source=var_dim(X_real,P_dec','dec');

% Method 9: Joint estimation from HS and MS 
% XM_deg = func_blurringY(XM,psfY);
% XM_deg = XM_deg(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
% Cov_M  = covariance_matrix(XM_deg,XM_deg);
% XHd_dec= var_dim(XHd,P_dec);
% Cov_HM = covariance_matrix(XHd_dec,XM_deg);
% %% Test the interpolation method
% % X_mean = imresize(XHd_dec,[nr,nc],'bicubic');
% % temp=Cov_HM/Cov_M*conv2mat(XM-imresize(XM_deg,[nr,nc],'bicubic'),size(XM,3));
% X_mean=ima_interp_spline(XHd_dec,psfY.ds_r);
% temp=Cov_HM/Cov_M*conv2mat(XM-ima_interp_spline(XM_deg,psfY.ds_r),size(XM,3));
% 
% [~,~,~,SNR_t,Q_t,SAM_t,RMSE_t,ERGAS_t,D_dist_t] = metrics(X_real,var_dim(X_mean,P_inc));
% fprintf('SNR: %f\n UIQI: %f\n SAM:  %f\n RMSE: %f\n ERGAS: %f\n D_dist:%f\n',SNR_t,Q_t,SAM_t,RMSE_t,ERGAS_t,D_dist_t);
% X_rough=X_mean+conv2im(temp,size(temp,1));

% VX_source=reshape(X_source,nr*nc,size(X_source,3))';
% X_source=repmat(mean(XM,3),[1 1 nb_sub]);
% X_source=var_dim(X_real,P_dec); % The source image is the real image prjected in subspace
%% Learn the dictionary
patsize = 6; 
tic 
%% DICTIONARY PARAMETERS
IsSubMean = 0; % remove mean ?
%% Coding parameters
maxAtoms = 4; % maximum number of atoms
%% For the moffet image 128*128 the maxAtoms is 8
%% For the Pavia  image 128*128 the maxAtoms is 5
delta = 1e-3; %  maximum  representation error in the dictionary. 
%Note: If this value is too small, the number of atoms in the coding is maxAtoms.
if learn_dic        
    %% LEARN  THE DICTIONARY
    [D,Ima_DL,out_para] = Dic_Learn(X_source,patsize,IsSubMean,method);
    if strcmp(method,'Cube')    
        D_band=reshape(D,nb_sub,size(D,1)/nb_sub,size(D,2));
        D_band=shiftdim(D_band,1);
    else
        D_band=D;
    end
    for i=1:size(D_band,3)
        figure(i);displayDic(D_band(:,:,i));
%         print(figure(i),'-depsc',strcat('D:\qwei2\Bayesian_fusion\figures\Dic_LMD',num2str(i)))
    end
    temp_name= ['Dic' num2str(nb_sub) '.mat'];
    save(temp_name,'D')
end
temp_name= ['Dic' num2str(nb_sub) '.mat'];load(temp_name);

if train_support
    Ima_DL=X_source;
    Xhat=zeros(size(Ima_DL));
    for i=1:size(Ima_DL,3)
        %             [Xhat,alpha] = compCode(Ima_DL(:,:,i), D(:,:,i), IsSubMean, maxAtoms, delta,method);
        [Xhat(:,:,i),alpha] = compCode(Ima_DL(:,:,i), D(:,:,i), IsSubMean, maxAtoms, delta,method);
        supp(:,:,i) = (alpha ~= 0);
        if numel(X_source)<Mem_lim
            if ~exist('D_s','var');
%                 D_s=zeros(patsize^2,patsize^2,size(alpha,2),nb_sub);
                D_s=zeros(patsize^2,patsize^2,size(alpha,2),size(Ima_DL,3));
            end
            for j=1:size(alpha,2)
                Daux = D(:,supp(:,j,i),i);
                D_s(:,:,j,i) = Daux*(Daux\eye(patsize^2));    
            end
        end
    end
    time_LD=toc;display(time_LD)
    if numel(X_source)<Mem_lim
        varargout{1}=D_s; %% Save the dictionary set
    else
        varargout{1}=D; 
        varargout{2}=supp;
    end   
    temp_name= ['supporte' num2str(nb_sub) '.mat'];save(temp_name,'supp','time_LD') %% Save the support
    for i=1:size(supp,3)
        figure(10);imagesc(supp(:,1500:1:2500,i));colormap gray;axis off;        %        set(gca,'YTick',[1:1:256]);
        set(gca,'Xtick',[]);
%         print(figure(10),'-depsc',strcat('D:\qwei2\Bayesian_fusion\figures\Supp_',num2str(i)))
    end
else
    temp_name= ['supporte' num2str(nb_sub) '.mat'];load(temp_name);
    if numel(X_source)<Mem_lim
        D_s=zeros(patsize^2,patsize^2,size(supp,2),nb_sub);
        for i=1:nb_sub
            for j=1:size(supp,2)
                Daux = D(:,supp(:,j,i),i);
                D_s(:,:,j,i) = Daux*(Daux\eye(patsize^2));   
            end
        end
        time_LD=toc;display(time_LD)
        varargout{1}=D_s;
    else
        varargout{1}=D;
        varargout{2}=supp;
        time_LD=toc;display(time_LD)
    end
end
%% Represent the MS image with learned dictionaries and support
if learn_dic + train_support >10
    %     X_test=var_dim(X_real,P_dec','dec');
    X_test=X_source;
    %   X_test=XM;
    X_restore=zeros(size(X_test));
    if strcmp(method,'Cube')
        for k=1:nb_sub
            PX_source(:,:,k) = im2col(X_test(:,:,k),[patsize patsize],'sliding');
        end
        temp=shiftdim(PX_source,2);
        temp=reshape(temp,size(temp,1)*size(temp,2),size(temp,3));
        % Restore the patches using dictionary and support
        for k=1:size(temp,2)
            Daux =  D(:,supp(:,k));
            Py_hat(:,k) = Daux*(Daux\temp(:,k));
        end
        clear tempY
        PYd=shiftdim(reshape(Py_hat,size(shiftdim(PX_source,2))),1);
        for k=1:size(PYd,3)
            X_restore(:,:,k)=depatch(PYd(:,:,k),patsize,XM(:,:,1));
        end
    else
        for k=1:size(X_test,3)
            if strcmp(method,'rotate')
                X_restore(:,:,k) = restoreFromSupp(X_test(:,:,k), D_s(:,:,:,k), IsSubMean);
                %                 X_restore(:,:,k) = restoreFromSupp(X_test(:,:,k), D, supp, IsSubMean);
            end
        end
    end
    [~,~,~,SNR_t,Q_t,SAM_t,RMSE_t,ERGAS_t,D_dist_t] = metrics(var_dim(X_test,P_inc),var_dim(X_restore,P_inc));
    fprintf('SNR: %f\n UIQI: %f\n SAM:  %f\n RMSE: %f\n ERGAS: %f\n D_dist:%f\n',SNR_t,Q_t,SAM_t,RMSE_t,ERGAS_t,D_dist_t);
    [~,~,~,SNR_t,Q_t,SAM_t,RMSE_t,ERGAS_t,D_dist_t] = metrics(X_real,var_dim(X_restore,P_inc));
    %     [err_max,err_l1,err_l2,Q,SAM,RMSE,ERGAS,D_dist] = metrics(X_real,var_dim(X_test,P_inc));
    fprintf('SNR: %f\n UIQI: %f\n SAM:  %f\n RMSE: %f\n ERGAS: %f\n D_dist:%f\n',SNR_t,Q_t,SAM_t,RMSE_t,ERGAS_t,D_dist_t);
    %     for i=1:size(X_test,3)
    %         figure;imagesc(X_restore(:,:,i)-X_test(:,:,i));colorbar;
    % %         figure;imagesc(X_test(:,:,i));colormap gray
    %         %figure;imagesc(supp(:,:,i));
    %     end
end