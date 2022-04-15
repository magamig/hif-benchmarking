function [P_tem,eig_val]=fac(Y)
% [P_tem,Eig_val]=eig(cov(reshape(Y,[size(Y,1)*size(Y,2) size(Y,3)])));
% [eig_val,Index]=sort(diag(Eig_val),'descend');
% N=max(N,3);
% P_vec=P_tem(:,Index(1:N)); % P_vec is an L*N matrix
temp=reshape(Y,[size(Y,1)*size(Y,2) size(Y,3)]);
[P_tem,eig_val,~]=svd(temp'*temp/size(temp,1));
eig_val=diag(eig_val);
% ratio_t=0;N=0;
% while ratio_t < ratio 
%     N=N+1;
%     ratio_t=sum(eig_val(1:N))/sum(eig_val);
% end
% P_vec=P_tem(:,1:N); % P_vec is an L*N matrix