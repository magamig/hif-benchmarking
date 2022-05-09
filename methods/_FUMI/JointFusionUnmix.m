function [Out,para_opt]= JointFusionUnmix(VXH,VXM,ChInv,CmInv,psfY,psfZ,sizeIM,para_opt_in)
%% This function implements the joint fusion and unmixing of Hyperspectral and Multispectral images
%If you use this code, please cite the following paper:
% [1] Q. Wei, J. M. Bioucas-Dias, N. Dobigeon and J-Y. Tourneret, 
%Multi-Band Image Fusion Based on Spectral Unmixing, in preparation.
%% -------------------------------------------------------------------------
% Copyright (March, 2015):        Qi WEI (qi.wei@n7.fr)
%
% FastAuxMat is distributed under the terms of
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
global E_real A_real X_real
N_it=para_opt_in.N_it;
SpaDeg=para_opt_in.SpaDeg;
thre_BCD=para_opt_in.thre_BCD;
E_hyper=para_opt_in.E_ini;
nr=sizeIM(1);nc=sizeIM(2);L=sizeIM(3); sizeMAT=[nr*nc L];
n_dr=nr/psfY.ds_r;     n_dc=nc/psfY.ds_r;
N_HS=size(VXH,1);   
% [H,~,~]=svd(E_real);
% H=H(:,1:L);

% th_h = 1e-4; % Threshold of change ratio in inner loop for HS unmixing
% th_m = 1e-4; % Threshold of change ratio in inner loop for MS unmixing
% sum2one = 1e-4;
% Out_Berne = NMF_Berne(nr,nc,psfY.ds_r,L,VXH,VXM,sum2one,2000,th_h,th_m,psfZ,E_VCA);
% E_hyper=Out_Berne.W_hyper(1:N_HS,:);
% E_VCA=E_hyper;
%% Estimate the endmembers in the subspace
% 
% scale=1;
%[H, P_dec, D, nb_sub]=idHSsub(X_real,'PCA',scale,L);
[H,~,~] = svd(E_real);
H=H(:,1:L);
% [H, P_dec, D, nb_sub]=idHSsub(reshape(VXH',[n_dr n_dc N_HS]),'PCA',scale,L);
% H=eye(N_HS);P_dec=H;
% [LE_VCA, ~, ~] = vca(VXM,'endmembers',L);
% E_VCA=H*(pinv(psfZ*H)*LE_VCA);%LH=psfZ*H;pinv(psfZ*H)=(LH'*LH)\LH'

if SpaDeg==1
    FBm   = fft2(psfY.B); 
    FBmC  = conj(FBm);
    FBs  = repmat(FBm,[1 1 L]);
    FBCs = repmat(FBmC,[1 1 L]);
    FBCNs= repmat(FBmC./(abs(FBmC).^2),[1 1 L]);
    B2Sum=PPlus(abs(FBs).^2./(psfY.ds_r^2),n_dr,n_dc);
    
    XH_int=zeros(nr,nc,N_HS);
    XH_int(1:psfY.ds_r:end,1:psfY.ds_r:end,:)=reshape(VXH',[n_dr n_dc N_HS]);
    VXH_int=reshape(XH_int,[nr*nc N_HS])';  %% interpolated HS image with zeros
    
    % Normalized HS and MS data
    NmVXH_int=ChInv*VXH_int;
    NmVXM=CmInv*VXM;
   
    rou=min(mean(diag(CmInv)),mean(diag(ChInv)))*1e0;
%     rou=1e5; % The ADMM regularization parameter
    meanVIm=zeros([L nr*nc]);  % The Gaussian prior mean: initialized with zeros
    W = zeros([L nr*nc]);      % Dual Variable: intialized with zeros
    N_admm=1e5;                % The maximum numbre of ADMM steps
%     thre_ADMM=1/sqrt(rou);            % The threshold of primal dual to stop in ADMM 1/sqrt(rou)
    thre_ADMM=1e-3;
    thre_ADMM_dual=1e-4;
    res_dual=inf;
    rou_set=zeros(1,N_admm);   % count the change of stepsize
    MSE_A_ADMM=zeros(1,N_admm);% MSE of each update in ADMM
end
cost_BCD=zeros(1,N_it);        % Initialization
vol_E=zeros(1,N_it); 
for t=1:N_it
    E_multi = psfZ*E_hyper; 
    if SpaDeg==0
      %% Update the HS Abundances: SUDAP or SUNSAL
      [A_hyper,~]= fcls_dpcs_v1(E_hyper,VXH,'POSITIVITY','yes','VERBOSE','no','ADDONE', 'yes', ...
                  'ITERS',100,'FACTORIZATION', 'orth', 'TOL', 1e-200,'X_SOL',0,'CONV_THRE',0);
      %% Update the MS Abundances: SUDAP or SUNSAL
      [A_multi,~]= fcls_dpcs_v1(E_multi,VXM,'POSITIVITY','yes','VERBOSE','no','ADDONE', 'yes', ...
                  'ITERS',100,'FACTORIZATION', 'orth', 'TOL', 1e-200,'X_SOL',0,'CONV_THRE',0); %% SUDAP
%       [A_multi,~,~,~] = sunsal(E_multi,VXM,'POSITIVITY','yes','VERBOSE','no','ADDONE', 'yes', ...
%                   'lambda', 0,'AL_ITERS',1e2,'TOL', 1e-200,'X_SOL',0,'CONV_THRE',0);          %% SUNSAL
    elseif SpaDeg==1
      %% Unmixing based on Sylvester equation embedded in ADMM      
      yyh   = E_hyper'*NmVXH_int;      
      yym   = E_multi'*NmVXM;
      ER1E  = E_hyper'*ChInv*E_hyper;
      RER2RE= E_multi'*CmInv*E_multi;
      %% Update the High-resolution Abundance Maps: ADMM scheme
%       [Q,Cc,InvDI,InvLbd,C2Cons]=FastAuxMat(yyh,yym,FBs,FBCs,sizeIM,n_dr,n_dc,rou*eye(L),ER1E,RER2RE,B2Sum);
      [Q,Cc,InvDI,InvLbd,C2Cons]=FasterAuxMat(yyh,yym,FBs,FBCs,sizeIM,n_dr,n_dc,rou*eye(L),ER1E,RER2RE,B2Sum);
      for n_admm=1:N_admm   
        %% Update A: Using An explicit Sylvester Solver         
%         VX_temp = FastFusion(meanVIm,Q,Cc,InvDI,InvLbd,C2Cons,B2Sum,FBs,FBCNs,sizeMAT,sizeIM,n_dr,n_dc,psfY.ds_r);
        VX_temp = FasterFusion(meanVIm,Q,Cc,InvDI,InvLbd,C2Cons,FBs,FBCs,sizeMAT,sizeIM,n_dr,n_dc,psfY.ds_r);
        A_multi = reshape(real(ifft2(reshape(VX_temp',sizeIM))),sizeMAT)';        
        %% Update V: Using the Projection onto the Simplex
        % Projection onto the ASC+ANC simplex: The complexity is O(nlogn) 
        %A_multi = SimplexProj(A_multi')';      
        V=SimplexProj((A_multi-W)')'; %  V=A_multi-W; Without projection to the simplex
        %% Update the residual
        res_pri=A_multi-V;  % The primal residual: the difference between A and V
        W=W-res_pri;        % The dual variable
        if n_admm>1
           res_dual=rou*(V-V_old);
%            [rou,W]=tune_penalty(rou,res_pri,res_dual,W);
        end           
        V_old=V;
        meanVIm = reshape(fft2(reshape((V+W)',sizeIM)),sizeMAT)';%VXd_dec VXHd_int VX_real Maybe problem here 
        if norm(res_pri,'fro')/sqrt(numel(res_pri))<thre_ADMM  %&& norm(res_dual,'fro')/sqrt(numel(res_dual))<thre_ADMM_dual   
%             disp(['ADMM converges at the ' num2str(n_admm) 'th iteration']);
            break; %% Stop when the primal dual is smaller than a threshold
        end
        rou_set(n_admm)=rou;         
        MSE_A_ADMM(n_admm)=norm(A_multi-A_real,'fro');
      end      
      ImAh_temp = reshape(A_multi',sizeIM);
      ImAh_temp = func_blurringY(ImAh_temp,psfY); % blurring and downsampling
      ImAh_temp = ImAh_temp(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
      A_hyper   = reshape(ImAh_temp,[n_dr*n_dc L])';
    end
    %% Update the Endmembers
     [E_hyper]= EEA(E_hyper,A_hyper,A_multi,VXH,VXM,psfZ,ChInv,CmInv,H); % HS+MS
%       E_hyper=E_VCA;
%     vol_E(t)=volume(E_hyper); % caluclate the volume of space spanned by E_hyper
%     E_hyper=min(max(E_hyper,0),1);
%     norm(E_hyper-E_real,'fro')
%     figure(1000);plot(E_hyper,'.');hold on;plot(E_real,'--');hold off
    % Display the restored image
%     X_FUMI=reshape((E_hyper*A_multi)',[nr nc N_HS]);
%     normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
%     temp_show=X_FUMI(:,:,[26 17 10]);temp_show=normColor(temp_show);
%     figure(116);imshow(temp_show);
%       [E_hyper]= EEA(E_hyper,A_hyper,A_multi,VXH,VXM,psfZ,ChInv,0); % Only HS    
    %% objective function to minimize
    cost_BCD(t)=norm(sqrt(ChInv)*(VXH-E_hyper*A_hyper),'fro')^2+norm(sqrt(CmInv)*(VXM-E_multi*A_multi),'fro')^2; 
    if t>2 && abs((cost_BCD(t)-cost_BCD(t-1))/cost_BCD(t-1))<thre_BCD
        disp(['BCD converges at the ' num2str(t) 'th iteration']);
        break;        
    end
%     disp(sum(sum((real(A_hyper_hat)-A_hyper).^2,1),2)); norm(E_LS-E_real,'fro') 
%     disp(norm(abs(E_hyper)-E_real,'fro'));
end
Out.E_hyper=E_hyper;
Out.A_hyper=A_hyper;
Out.E_multi=E_multi;
Out.A_multi=A_multi;

para_opt.cost_BCD=cost_BCD;
para_opt.vol_E=vol_E;
para_opt.H=H;
% para_opt.P_dec=P_dec;
if SpaDeg==1
    para_opt.rou_set=rou_set;
end