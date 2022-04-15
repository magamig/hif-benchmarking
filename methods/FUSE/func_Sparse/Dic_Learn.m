function [D,Ima_DL,varargout] = Dic_Learn(Xp,patsize,IsSubMean,method)
% take patches
for i=1:size(Xp,3)
    PXp(:,:,i) = im2col(Xp(:,:,i),[patsize patsize],'sliding');
end
%%%%%%%%%%%%%%%%%%%%%%%%input parameters for dictionary learning%%%%%%%%%%%
%  param.k                    number of atoms in the trained dictionary
%                             defaul: 256
%  param.t                    number of iterations in the trained dictionary
%                             defaul: 500
%  param.patchnum             number of patches use in one ieration in the trained dictionary
%                             defaul: 64
%  param.lamda                parameter in sunsal to solve the
%                             defaul: 1.8
%  param.err                  tolerant error in the sunsal algorithm
%                             defaul: 1e-2  
%%%%%%%%%%%%%%%%%%%%%%%output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.K = 256;
param.T = 500;
param.patchnum = 64;
param.lamda = 0.11;
param.patsize = patsize;
param.err = 1e-2;

% remove mean
if IsSubMean == 1
    %         PXp_all=[];
    for i=1:size(PXp,3)
        meancolx = mean(PXp(:,:,i));
        PXp(:,:,i) = PXp(:,:,i)-repmat(meancolx,[size(PXp,1) 1]);
        %             PXp_all= [PXp_all; PXp(:,:,i)];
    end
end

if strcmp(method,'Pan')
    % Method 1: Lean Dic from Pan_like image
    Ima_DL=mean(Xp,3);
    temp = im2col(Ima_DL,[patsize patsize],'sliding');
    D = DicLearningM(temp,param);
%     ratio=0.75;[P_vec,~]=fac(Xp,ratio);P_vec=P_vec(:,1);
%     Ima_DL=var_dim(Xp,P_vec,'dec');

    varargout{1}=[];
elseif strcmp(method,'PCA')
    % Method 2: Lean Dic from subspace of MS image 
    [P_vec,~]=fac(Xp);P_vec=P_vec(:,1);
    Ima_DL=var_dim(Xp,P_vec,'dec');
    temp = im2col(Ima_DL,[patsize patsize],'sliding');
    D = DicLearningM(temp,param);
    varargout{1}=[];
elseif strcmp(method,'Batch')
    temp=reshape(PXp,size(PXp,1),size(PXp,2)*size(PXp,3));
    D = DicLearningM(temp,param);%128
    
    ratio=0.75;[P_vec,~]=fac(Xp,ratio);P_vec=P_vec(:,1);
    Ima_DL=var_dim(Xp,P_vec,'dec');
%     Ima_DL=mean(Xp,3);
    varargout{1}=[];
elseif strcmp(method,'BbB')
    % Submethod 1: Lean Dic from MS image 
%     for i=1:size(PXp,3)
%         param.K = 64;D(:,(i-1)*param.K+1:i*param.K) = DicLearningM(PXp(:,:,i),param);
%     end
%     varargout{1}=[];
%     ratio=0.75;[P_vec,~]=fac(Xp,ratio);P_vec=P_vec(:,1);
%     Ima_DL=var_dim(Xp,P_vec,'dec');
    
    % Submethod 2: Lean Dic from MS image
    D=zeros(patsize^2,param.K);
    param.K = 160;                
    Set = 1:param.K;  out_para.N_ave=param.K;
    
    D(:,Set) = DicLearningM(mean(PXp,3),param);
    param.K = (size(D,2)-out_para.N_ave)/size(Xp,3);
    out_para.N_band=param.K;
                
    Set=out_para.N_ave-out_para.N_band+1:out_para.N_ave; 
    
    out_para.SetIni=Set;
    
    varargout{1}=out_para;
    for i=1:size(Xp,3)
        Set = Set+param.K; D(:,Set) = DicLearningM(PXp(:,:,i)-mean(PXp,3),param);
    end
    
    [P_vec,~]=fac(Xp);P_vec=P_vec(:,1);
    Ima_DL=var_dim(Xp,P_vec,'dec');
elseif strcmp(method,'rotate') || strcmp(method,'ms_sub') %% The active methdod
    D=zeros(patsize^2,param.K,size(PXp,3));
    for i=1:size(PXp,3)
        D(:,:,i) = DicLearningM(PXp(:,:,i),param);
    end
    Ima_DL=Xp;
    varargout{1}=[];
elseif strcmp(method,'Cube')
    temp=shiftdim(PXp,2);
    Cube_patch=reshape(temp,size(temp,1)*size(temp,2),size(temp,3));
    D = DicLearningM(Cube_patch,param);
    Ima_DL=Cube_patch;
    varargout{1}=[];
end