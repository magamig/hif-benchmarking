% Alternating Optimization algorithm
% define and initialize variables
% Z:      the target in image form 
% Z_dec:  the subspace transformed subspace in image form 
function [Zim,Costime,diff_X,RMSE_sub,RMSE_org,varargout]=AlterOpti(X_rough,XH,XM,psfY,psfZ,s2y_real,s2z_real,P_dec,P_inc,FusMeth,X_real,varargin)
[nr,nc,~]=size(XM); % The size of the MS data
nb_sub=size(P_dec,1); % The dimension of subspace or the number of principal components
N_band=size(XH,3);   % The dimension of HS dat
[VXM,VXH,FBm,FBmC,FZ,FZ_s,FZC,FZC_s,mask_s,ConvCBD,conv2im,conv2mat] = func_define(XM,XH,psfY,nb_sub);
% psfZ=psfZ_estimate(XHd,XM,psfY);
% psfZ=psfZ./repmat(sum(psfZ,2),1,size(psfZ,2));
% psfZ=psfZ_unk;
psfZ_s=psfZ*P_inc;% size:4*6
% VXHd_int=reshape(var_dim(XHd_int,P_dec),nr*nc,nb_sub)';
VXHd_int=reshape(ima_interp_spline(var_dim(XH(1:psfY.ds_r:end,1:psfY.ds_r:end,:),P_dec),psfY.ds_r),nr*nc,nb_sub)';
VX_real=reshape(var_dim(X_real,P_dec),nr*nc,nb_sub)';
%--------------------------------------------------------------------------
%   Corrupt with additive noise 
%--------------------------------------------------------------------------
R1 = zeros(N_band); R2 = zeros(size(XM,3));
for i=1:N_band;            R1(i,i) = 1/s2y_real(i);   end;  % HS bands
for i=1:size(VXM,1);       R2(i,i) = 1/s2z_real(i);   end;  % MS bands 
%% The regularization parameter
% tau_begin=0; tau_stop=50;% Moffet
% tau_begin=30; tau_stop=100;% Pavia
% tau_begin=15; tau_stop=35;% Pavia whole
% step=(tau_stop-tau_begin)/5;
% tau_d_set=tau_begin:step:tau_stop;
tau_d_set=25;% Pavia  HS+mS
% tau_d_set=75; %Moffet HS+MS
% tau_d_set=80;% Moffet HS+PAN
% tau_d_set=18;% Pavia HS+MS whole image
Zim=zeros([size(X_real) length(tau_d_set)]);
for j=1:length(tau_d_set)
if strcmp(FusMeth,'Sparse') || strcmp(FusMeth,'BCG')
    II_FB = repmat(1./(FBm.*FBmC + 2),[1 1 N_band]);II_FB_s = II_FB(:,:,1:nb_sub);
    if strcmp(FusMeth,'Sparse')
%         Para_Prior=varargin{1};
        if size(varargin{1},4)>1         
            D_s=varargin{1};
        else
            Dic=varargin{1}; 
            supp=varargin{2};
        end
        IsSubMean=0;
        tau_d=tau_d_set(j);
    elseif strcmp(FusMeth,'BCG')
        Para_Prior=varargin{1}; Prior_Ini; tau_d=1;
    end
elseif strcmp(FusMeth,'GMM')
    obj=varargin{1};        obj_est=varargin{2};
    pat_mode=varargin{3};     N_pat=varargin{4};
    shift_size=varargin{5};
    
    patsize=sqrt(size(obj.mu,1));
    N_ga=length(obj.alpha);
    Cube=1;
    if strcmp(pat_mode,'distinct')
        II_FBGM=repmat(1./(FBm.*FBmC + 2),[1 1 N_band]);
    elseif strcmp(pat_mode,'sliding')
        II_FBGM=repmat(1./(FBm.*FBmC + patsize^2 + 1),[1 1 N_band]);
    end
    II_FBGM_s = II_FBGM(:,:,1:nb_sub);
    tau_d=1;
end
%% The Lagrange parameter
%  mu = 0.05;% mu_set = 0.01:1:10.01;
% mu = 5e-2/mean(s2y_real); %% For patsize=1;
mu = 5e-2/mean(s2y_real);
% for mu=mu_set
tic
%% Initialization
%     epsilon1=1e2*ones(1,N_band);
%     epsilon2=1e2*ones(1,size(XM,3));
% VX_dec=VXHd_int; % Initialization of X    % VX_dec=VX_dec_ref;
if strcmp(FusMeth,'Sparse') || strcmp(FusMeth,'GMM')
    VX_dec=reshape(X_rough,[nr*nc nb_sub])';
else
    VX_dec=VXHd_int;
end
RMSE_sub_ini=mean2((VX_dec-VX_real).^2);
RMSE_org_ini=mean2((P_inc*(VX_dec-VX_real)).^2);
display([RMSE_sub_ini RMSE_org_ini])
if strcmp(FusMeth,'GMM')
    % Do the math in advance to accelerate the speed: band-wise and Gaussian-term-wise
    InvCovGM=zeros(size(obj.Cov)); % The inverse of covariance matrix
    InvCovMu=zeros(size(obj.Cov)); % The inverse of Cov+mu*I
    InvCovTem = zeros([patsize^2 1 N_ga nb_sub]);
    for i=1:nb_sub
        for k=1:N_ga
            InvCovGM(:,:,k,i)=inv(obj.Cov(:,:,k,i));
            InvCovMu(:,:,k,i)=inv(InvCovGM(:,:,k,i)+mu*eye(patsize^2));
            InvCovTem(:,:,k,i)=InvCovGM(:,:,k,i)*obj.mu(:,k,i);
%             InvCovTem(:,:,k,i)=mtimesx(InvCovGM(:,:,k,i),reshape(obj.mu(:,k,i),[patsize^2 1 1 1]));
        end
    end
    label=LabelUpdate(VX_dec,VXM,N_pat,obj,obj_est,conv2im,pat_mode,shift_size,Cube); %VX_real
    figure(80);imagesc(col2im(repmat(label,[patsize^2 1]),[patsize patsize],[nr nc],'distinct'));
end

V1 = ConvCBD(VX_dec,FZ_s);  G1 = V1;
V2 = VX_dec;                G2 = V2;
if strcmp(FusMeth,'BCG') || strcmp(FusMeth,'Sparse')
    V3 = VX_dec; G3 = VX_dec;
    VXd_dec = VX_dec; % VXd_dec: The prior mean of VX_dec (the target image)
    if strcmp(FusMeth,'BCG') 
        invCov_U=Para_Prior.invCov_U0; % Updated in the optimization
    elseif strcmp(FusMeth,'Sparse')
%         invCov_U=Para_Prior.invCov_U0;
        invCov_U=eye(nb_sub);          % Fixed in the optimization
    end
elseif strcmp(FusMeth,'GMM')  
    MX_dec=conv2im(VX_dec,nb_sub);
    V3=zeros([patsize^2 1 N_pat nb_sub]);
    for i=1:nb_sub
        if strcmp(pat_mode,'sliding')
            temp=padarray(MX_dec(:,:,i),[patsize-1 patsize-1],'circular','post');
        elseif strcmp(pat_mode,'distinct')
            temp=circshift(MX_dec(:,:,i),shift_size);
        end
        temp=im2col(temp,[patsize patsize],pat_mode);
        V3(:,:,:,i) = reshape(temp,[patsize^2 1 N_pat 1]);
    end
    G3 = V3;
    %         G3 = V3*0; 
end
% All variables (Z,V1,V2,V3,G1,G2,G3)  are matrices
%% ADMM parameters
iters = 1e1;   %Iters for the external loop    
ADMM='on';admm_iters = 50; %tol_cost=1e-5;cost_fun=zeros(1,iters);temp3=zeros([nr nc nb_sub]);
if strcmp(FusMeth,'Sparse'); tol_para=3e-3; %tol_para=1e-2;
else    tol_para=1e-2;
end
tau_daux = tau_d;
RMSE_sub=zeros(1,iters);RMSE_org=zeros(1,iters);diff_X=zeros(1,iters);
%% Alternating optimization
for t=1:iters
    %        if strcmp(FusMeth,'BCG') || strcmp(FusMeth,'Sparse')
    %            cost_fun(t)=tau_daux/2*trace((VX_dec-VXd_dec)*(VX_dec-VXd_dec)
    %            '*invCov_U); %% Note trace(x^T \Sigma x)= trace(x x^T \Sigma) 
    %        elseif strcmp(FusMeth,'GMM')  
%            cost_fun(t)=0;
%            for i=1:nb_sub
%                if strcmp(pat_mode,'sliding')
%                    temp=padarray(MX_dec(:,:,i),[patsize-1 patsize-1],'circular','post');
%                elseif strcmp(pat_mode,'distinct')
% %                    temp=MX_dec(:,:,i);
%                      if shift==1
%                          temp=circshift(MX_dec(:,:,i),shift_size);
%                      end                   
%                end
%                temp=im2col(temp,[patsize patsize],pat_mode)-obj.mu(:,label(i,:),i);
%                temp_cost=mtimesx(obj.Cov(:,:,label(i,:),i),reshape(temp,[patsize^2 1 N_pat]));
%                cost_fun(t)=cost_fun(t)+sum(sum(temp.*reshape(temp_cost,[patsize^2 N_pat]).*repmat(obj.alpha(label(i,:),i)',[patsize^2 1])))/2;
%            end
%        end
%        cost_fun(t)=cost_fun(t)+1/2*norm(sqrt(R1)*(VXH-mask.*ConvCBD(P_inc*VX_dec,FZ)),'fro')^2+...
%                                1/2*norm(sqrt(R2)*(VXM-psfZ_s*VX_dec),'fro')^2;

VX_dec_old=VX_dec;
% if t>1; diff_X_old=diff_X;end
if strcmp(ADMM,'on')
    % SALSA  (ADMM) algorithm
    for i=1:admm_iters
        %% Update U: min_U ||UB-V1-G1||_F^2 +||U-V2-G2||_F^2 +||U-V3-G3||_F^2     
        if strcmp(FusMeth,'BCG') || strcmp(FusMeth,'Sparse')  
            VX_dec = ConvCBD(ConvCBD(V1+G1,FZC_s) + (V2+G2) + (V3+G3),II_FB_s);
        elseif strcmp(FusMeth,'GMM')                   
%             temp=(V3+G3);
%             if strcmp(pat_mode,'distinct')
%                 for k=1:nb_sub 
%                     temp3(:,:,k)=col2im(reshape(temp(:,:,:,k),[patsize^2 N_pat 1]),[patsize patsize],[nr nc],'distinct');                
%                 end
%                 temp3=circshift(temp3,shift_size*(-1));
%             elseif strcmp(pat_mode,'sliding')
%                 for k=1:nb_sub 
%                     %                         [temp3(:,:,k),D_est]=depatch(reshape(temp(:,:,:,k),[patsize^2 N_pat 1]),patsize,X_real(:,:,1));
% %                         temp3(:,:,k)=temp3(:,:,k).*D_est;
%                     temp3(:,:,k)=col2im(reshape(temp(1,:,:,k),[1 N_pat]),[patsize patsize],[nr+patsize-1 nc+patsize-1],pat_mode);
%                     temp3(:,:,k)=temp3(:,:,k)*patsize^2;               
%                 end
%             end

%             temp3= reshape(V3+G3,[nr nc nb_sub]);% Simple case 
%             VX_dec = ConvCBD(ConvCBD(V1+G1,FZC_s) + (V2+G2) + conv2mat(temp3,nb_sub),II_FBGM_s);
            VX_dec = ConvCBD(ConvCBD(V1+G1,FZC_s) + (V2+G2) + (squeeze(V3+G3))',II_FBGM_s);
        end
        MX_dec = conv2im(VX_dec,nb_sub);
       %% Update V1
        % min_{V1} (1/2) ||Yr-R(M(V1))|_F^2  + (mu/2)||B(Z) - V1 - G1||_F^2 
        NU1 = ConvCBD(VX_dec,FZ_s) - G1;
        V1 = ((P_inc'*R1*P_inc)+mu*eye(nb_sub))\(P_inc'*R1*VXH+mu*NU1);
        V1_com = (1-mask_s).*NU1; 
        V1 = V1.*mask_s+V1_com;
        % norm(R*(Y-(ConvCBD(Z,FZ).*mask)),'fro') % show the data misfit term
       %% Update V2
        % min_{V2} (tau_w/2)  ||WV2||_F^2 + (mu/2)||Z   - V2 - G2||_F^2
        NU2 = VX_dec-G2;
        V2 = (psfZ_s'*R2*psfZ_s+mu*eye(nb_sub))\(psfZ_s'*R2*VXM+mu*NU2);
        %% Update V3
        % min_{V3} (tau_daux/2) ||V3 - VXd_dec||_F^2 + (mu/2)||P_dec*Z  - V3 - G3||_F^2
        if strcmp(FusMeth,'BCG') || strcmp(FusMeth,'Sparse')
            NU3 = VX_dec-G3;
            V3 = (tau_daux*invCov_U+mu*eye(nb_sub))\(tau_daux*invCov_U*VXd_dec + mu*NU3);
            G3 = -NU3 + V3; 
        elseif strcmp(FusMeth,'GMM')
            %% Patches 
            for k=1:nb_sub
%                 if strcmp(pat_mode,'sliding')
%                     temp = padarray(MX_dec(:,:,k),[patsize-1 patsize-1],'circular','post');
%                 elseif strcmp(pat_mode,'distinct')
%                     temp = circshift(MX_dec(:,:,k),shift_size);
%                 end
%                 NU3 = reshape(im2col(temp,[patsize patsize],pat_mode),[patsize^2 1 N_pat])-G3(:,:,:,k);

                NU3 =reshape(MX_dec(:,:,k),[1 1 N_pat])-G3(:,:,:,k); % Simple Case
                
                if Cube==0
                    V3_tempA=InvCovTem(:,:,label(k,:),k)+mu*NU3;
                    V3_tempB=InvCovMu(:,:,label(k,:),k);
                elseif Cube==1                           
                    V3_tempA=InvCovTem(:,:,label,k)+mu*NU3; %% The label for all the bands are same
                    V3_tempB=InvCovMu(:,:,label,k); %% The label for all the bands are same
                end
                %                         V3(:,:,:,k)= NU3;
%                 tic;V3(:,:,:,k)=mtimesx(V3_tempB,V3_tempA);toc;
                V3(:,:,:,k)=V3_tempB.*V3_tempA;
                G3(:,:,:,k)= -NU3 + V3(:,:,:,k);  
            end
        end
        %fprintf('iter = %d, e1 = %2.2f, e2 = %2.2f, e3 = %2.2f\n',i,norm(NU1+G1-V1, 'fro'),norm(NU2+G2-V2, 'fro'),norm(NU3+G3-V3, 'fro'))
        % update Lagrange multipliers
        G1 = -NU1 + V1;    % G1 -(ZBm-V1)
        G2 = -NU2 + V2;    % G2 -(ZBp-V2)
    end
    %             Zim = conv2im(P_dec'*VX_dec,N_band);
    %             RMSE_temp(t)=sqrt(mean2((Zim - X_real).^2));
elseif strcmp(ADMM,'off')
    tol_CG=1e-6;maxit_CG=10;stopping_rule=1;verbose_CG='off';
    %             AA = @(x_sub)(P_dec*ConvCBD(mask.*ConvCBD(P_dec'*x_sub,FZ),FZC)*(R1)+...
%                 P_dec*(psfZ'*psfZ)*P_dec'*x_sub*(R2) + tau_d*x_sub);
%             bb = tau_d*VXd_dec + R1^2*P_dec*ConvCBD(mask.*VXH,FZC)+R2^2*P_dec*psfZ'*VXp;
    AA = @(x_sub)(P_dec*R1*ConvCBD(mask.*ConvCBD(P_dec'*x_sub,FZ),FZC)+...
        P_dec*(psfZ'*R2*psfZ)*P_dec'*x_sub + tau_d*(psfZ_s'*psfZ_s)*x_sub);
    bb = tau_d*psfZ_s'*VXd_dec + P_dec*R1*ConvCBD(mask.*VXH,FZC)+P_dec*psfZ'*R2*VXp;
    [VX_dec,flag,iter,resvec,func] = ConjGradient2D(AA,bb,[],tol_CG,maxit_CG,VX_dec,stopping_rule,verbose_CG);
    iter
end
if strcmp(FusMeth,'BCG')
    %% Update the covariance matrix
    if t~=0
        tempX=VX_dec-VXd_dec;
%         tempX=VX_real-VXd_dec;
        invCov_U=inv((tempX*tempX'+Phi)/(size(tempX,2)+nb_sub+eta+1));
    end
    %% Update the HS noise covariances
%         tempY=conv2im(P_inc*VX_dec,size(P_dec,2));
%         tempY=XH-func_blurringY(tempY,psfY);  
%         tempY=tempY(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
%         for i=1:N_band 
%             tempY_b=tempY(:,:,i); 
%             if t~=0
%                 R1(i,i) = (numel(tempY_b)+nuH+2)/(norm(tempY_b,'fro')^2+gammaH(i));   
%             else
%                 beta = norm(tempY_b,'fro')^2/2;
%                 alpha= numel(tempY_b)/2;
%                 %             grad=grad_VarNoise(R1(i,i),alpha,beta);
%                 f_cost = @(x) -(alpha+1)*log(x)+beta*x;
%                 f_grad = @(x) -(alpha+1)/x+beta;        
%                 for index=1:10
%                     temp=f_grad(R1(i,i));
%                     epsilon1(i) = armijo(epsilon1(i),R1(i,i),(-1)*temp,f_cost,f_grad);
%                     R1(i,i) = R1(i,i)-epsilon1(i)*temp;
%                 end
%             end
%         end  % HS bands
        %% Update the MS noise covariances
% %         Ps = squeeze(mean(mean(VX_real.^2,1),2));
% %         sigma2 = Ps.*(10.^(-26/10));     VX_test=VX_real+randn(size(VX_real))*sqrt(sigma2);
%         tempZ=VXM-psfZ_s*VX_dec;
%         %         tempZ=conv2im(tempZ,size(tempZ,1));
% %         tempZ=tempZ(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
% %         tempZ=reshape(tempZ,size(tempZ,1)*size(tempZ,2),size(tempZ,3))';
%         for j=1:size(VXM,1)
%             tempZ_b=tempZ(j,:);  
%             if t~=0
%                 R2(j,j) = (numel(tempZ_b)+nuM+2)/(norm(tempZ_b,'fro')^2+gammaM(j)); 
% %                 R3(j,j) = (numel(tempZ_b)+2)/(norm(tempZ_b,'fro')^2); 
%             else
%                 beta = norm(tempZ_b,'fro')^2/2;
%                 alpha= numel(tempZ_b)/2;
%                 %             temp=grad_VarNoise(R2(j,j),alpha,beta);
%                 
%                 f_cost = @(x) -(alpha+1)*log(x)+beta*x;
%                 f_grad = @(x) -(alpha+1)/x+beta;     
%                 for index=1:10
%                     temp=f_grad(R2(j,j));
%                     epsilon2(j) = armijo(epsilon2(j),R2(j,j),(-1)*temp,f_cost,f_grad);
%                     R2(j,j) = R2(j,j)-epsilon2(j)*temp;
%                 end
%             end
%         end  % MS bands
% % %         figure(520);plot(diag(R2));
end
%% Update the hidden variable
if strcmp(FusMeth,'BCG')
    VXd_dec=VXHd_int;     %use the interpolated image as constraint     
elseif strcmp(FusMeth,'GMM')
    label=LabelUpdate(VX_dec,VXM,N_pat,obj,obj_est,conv2im,pat_mode,shift_size,Cube);
    figure(80);imagesc(col2im(repmat(label,[patsize^2 1],[]),[patsize patsize],[nr nc],'distinct'));
elseif strcmp(FusMeth,'Sparse')
    % A-Step using OMP: Compute the L(DA) band by band
    for k=1:nb_sub
        if size(varargin{1},4)>1 
            Im = restoreFromSupp(conv2im(VX_dec(k,:),1), D_s(:,:,:,k));
        elseif size(varargin{1},4)==1 
            Im = restoreFromSupp(conv2im(VX_dec(k,:),1), Dic(:,:,k),supp(:,:,k));
        end
        VXd_dec(k,:) = conv2mat(Im,1);
    end
end
%% Stopping rule
diff_X(t)=norm(VX_dec-VX_dec_old,'fro')/norm(VX_dec_old,'fro');
%% Evaluate the iterative estimation                           
RMSE_sub(t)=mean2((VX_dec-VX_real).^2);
RMSE_org(t)=mean2((P_inc*(VX_dec-VX_real)).^2);
%           diff_cost=(cost_fun(t)-cost_fun(t-1))/cost_fun(t-1);
%     diff2_X=norm(diff_X-diff_X_old,'fro')/norm(diff_X_old,'fro');
%           display([diff_cost diff_Z_dec])
display([diff_X(t) RMSE_sub(t) RMSE_org(t) ])
if abs(diff_X(t))<tol_para && t~=1 && t>2 % abs(diff_cost) < tol_cost  %|| t==2
    diff_X=diff_X(diff_X~=0);
    RMSE_sub=RMSE_sub(RMSE_sub~=0);
    RMSE_org=RMSE_org(RMSE_org~=0);
    break
end
admm_iters = 20;
end
%% After the optimization:
VX_est=P_inc*VX_dec;
Zim(:,:,:,j) = conv2im(VX_est,N_band);  Costime.(FusMeth)=toc;
end
%% Output the hidden variable
if strcmp(FusMeth,'GMM');varargout{1}=label;
elseif strcmp(FusMeth,'BCG');varargout{1}=invCov_U;
elseif strcmp(FusMeth,'Sparse');varargout{1}=tau_d_set;varargout{2}=VXd_dec;
end
figure(111);%figure(gcf);drawnow;
% subplot(2,1,1);semilogy(cost_fun); xlabel('iterations');title('Cost function')
subplot(2,1,1);semilogy(RMSE_sub);xlabel('iterations');title('RMSE of estimation in subspace');
subplot(2,1,2);semilogy(RMSE_org); xlabel('iterations');title('RMSE of estimation in original space');