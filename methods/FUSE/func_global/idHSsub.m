%% HS subspace identification: Identifying the subspace where HS data live in
function [P_inc, P_dec, D, nb_sub]=idHSsub(XHd,subMeth,scale)
temp=reshape(XHd,[size(XHd,1)*size(XHd,2) size(XHd,3)])';
% temp=temp-repmat(mean(temp,2),[1,size(temp,2)]); % or equally temp=bsxfun(@minus, temp, mean(temp,2));
if strcmp(subMeth,'Hysime')
    [w,Rn] = estNoise(temp);
    [nb_sub,P_vec,eig_val]=hysime(temp,w,Rn);%diag(sigma2y_real)
elseif strcmp(subMeth,'PCA')
    [P_vec,eig_val]=fac(XHd);
    %[w,Rn] = estNoise(temp);    [nb_sub,~,~]=hysime(temp,w,Rn);%diag(sigma2y_real)
    nb_sub=15;
    PCA_ratio=sum(eig_val(1:nb_sub))/sum(eig_val); %[P_vec,eig_val]=fac(X_real);
    P_vec=P_vec(:,1:nb_sub); % Each column of P_vec is a eigenvector
end
if scale==1
    D=diag(sqrt(eig_val(1:nb_sub)));
    P_dec=D\P_vec';
    P_inc=P_vec*D;
elseif scale==0
    D=0;
    P_dec=P_vec';
    P_inc=P_vec;
end