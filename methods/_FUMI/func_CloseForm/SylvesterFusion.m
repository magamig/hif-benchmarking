function [VX]=SylvesterFusion(VXH_int,VXM,psfY,psfZ,ChInv,CmInv,E_hyper,Xd_dec,sizeIM,N_AO,prior)
%% function SylvesterFusion is designed for demo purpose (1.IEEEGRSM paper 2.Airbus dataset)
nr=sizeIM(1);
nc=sizeIM(2);
L =sizeIM(3);
sizeMAT=[nr*nc L];

n_dr=nr/psfY.ds_r;
n_dc=nc/psfY.ds_r;
E_multi = psfZ*E_hyper; 
%Auxiliary Matrices
FBm   = fft2(psfY.B); 
FBmC  = conj(FBm);
FBs  = repmat(FBm,[1 1 L]);
FBCs = repmat(FBmC,[1 1 L]);
FBCNs= repmat(FBmC./(abs(FBmC).^2),[1 1 L]);
B2Sum=PPlus(abs(FBs).^2./(psfY.ds_r^2),n_dr,n_dc);

% Normalized HS and MS data
NmVXH_int= ChInv*VXH_int;
NmVXM    = CmInv*VXM;
yyh      = E_hyper'*NmVXH_int;      
yym      = E_multi'*NmVXM;
ER1E     = E_hyper'*ChInv*E_hyper;
RER2RE   = E_multi'*CmInv*E_multi;

if strcmp(prior,'Gaussian')
%     invCov = 3e-2*eye(L); %% Airbus
%     invCov = 1e-1*eye(L); %% Airbus VCA ( or no subspace): 1e-1 PCA:1e-3  
%     [~,eig_val,~]=svd(VXH_int*VXH_int'/size(VXH_int,2)*psfY.ds_r*psfY.ds_r);
%     invCov = eig_val*eye(L)*1e-1; % it is equivalent to assign prior in orignial space
    invCov = 1e-3*eye(L);
elseif strcmp(prior,'ML')
    invCov = 1e-17*eye(L); %% ML ratio 1e-13
end
% invCov = 1e-1*eye(L); %% Airbus
% [Q,Cc,InvDI,InvLbd,C2Cons]=FastAuxMat(yyh,yym,FBs,FBCs,sizeIM,n_dr,n_dc,invCov,ER1E,RER2RE,B2Sum);
[Q,Cc,InvDI,InvLbd,C2Cons]=FasterAuxMat(yyh,yym,FBs,FBCs,sizeIM,n_dr,n_dc,invCov,ER1E,RER2RE,B2Sum);
if strcmp(prior,'Sparse')
    load('Dic_Learn.mat');
end
for i=1:N_AO 
    meanVIm = reshape(fft2(Xd_dec),sizeMAT)';%VXd_dec VXHd_int VX_real Maybe problem here 
%     VX_FFT  =   FastFusion(meanVIm,Q,Cc,InvDI,InvLbd,C2Cons,B2Sum,FBs,FBCNs,sizeMAT,sizeIM,n_dr,n_dc,psfY.ds_r);
    VX_FFT  = FasterFusion(meanVIm,Q,Cc,InvDI,InvLbd,C2Cons,FBs,FBCs,sizeMAT,sizeIM,n_dr,n_dc,psfY.ds_r);
    VX=reshape(real(ifft2(reshape(VX_FFT',sizeIM))),sizeMAT)';
    % A-Step using OMP: Compute the L(DA) band by band
    if strcmp(prior,'Sparse')
        for k=1:L
            Xd_dec(:,:,k) = restoreFromSupp(reshape(VX(k,:)',[nr nc]), D_s);
    %         Im = restoreFromSupp(reshape(VX(k,:)',[nr nc]), D,supp);
        end    
    end
end