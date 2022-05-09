%% HS subspace identification: Identifying the subspace where HS data live in
function [P_inc, P_dec, D, nb_sub]=idHSsub(XHd,subMeth,scale,varargin)
temp=reshape(XHd,[size(XHd,1)*size(XHd,2) size(XHd,3)])';
% temp=temp-repmat(mean(temp,2),[1,size(temp,2)]); % or equally temp=bsxfun(@minus, temp, mean(temp,2));  
if nargin == 4
    nb_sub=varargin{1};
end
if strcmp(subMeth,'Hysime')
    [w,Rn] = estNoise(temp);
    [nb_sub,P_vec,eig_val]=hysime(temp,w,Rn);%diag(sigma2y_real)
    eig_val=sqrt(eig_val); %% Note that the square root is required as the eig_val is the eigen value of X'*X;
%     nb_sub=5;
    P_vec=P_vec(:,1:nb_sub);
elseif strcmp(subMeth,'PCA')
    [P_vec,eig_val]=fac(XHd);
    %[w,Rn] = estNoise(temp);    [nb_sub,~,~]=hysime(temp,w,Rn);%diag(sigma2y_real)
    PCA_ratio=sum(eig_val(1:nb_sub))/sum(eig_val); %[P_vec,eig_val]=fac(X_real);
    P_vec=P_vec(:,1:nb_sub); % Each column of P_vec is a eigenvector
elseif strcmp(subMeth,'VCA')
    %     Find endmembers with VCA (pick the one with smallest volume from 20 
%     runs of the algorithm)
    max_vol = 0;
    vol = zeros(1, 20);
    for idx_VCA = 1:20
        E_aux = VCA(temp,'Endmembers',nb_sub,'SNR',0,'verbose','off');
        vol(idx_VCA) = abs(det(E_aux'*E_aux));
        if vol(idx_VCA) > max_vol
            P_vec = E_aux;
            max_vol = vol(idx_VCA);
        end   
    end
end
if scale==1
%     D=diag(sqrt(eig_val(1:nb_sub))); %% Note that there is no square root
    D=diag(eig_val(1:nb_sub));
% %     here.( D=diag(sqrt(eig_val(1:nb_sub))) is wrong!!!)
%     D=diag(eig_val(1:nb_sub)); 
    P_dec=D\P_vec';
    P_inc=P_vec*D;
elseif scale==0
    D=eye(nb_sub);
    P_dec=P_vec';
    P_inc=P_vec;
end