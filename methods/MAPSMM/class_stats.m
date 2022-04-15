function [mj_pure,Cj_pure]=class_stats(u,cmap)
%
% u (nk+nm)*hx*hy
% cmap hx*hy
%
% mj_pure (nk+nm)*nq
% Cj_pure nq*(nk+nm)*(nk*nm)
[hx,hy]=size(cmap);
nq=max(cmap(:));
nknm=size(u,1);
%cmap2=cmap;
%cmap=permute(reshape(repmat(cmap(:),nknm,1),hx,hy,nknm),[3 1 2]);
mj_pure=zeros(nknm,nq);
Cj_pure=zeros(nq,nknm,nknm);

for q=1:nq
    %q_num=sum(sum(cmap2==q));
    %mj_pure(:,q)=sum(sum(u.*(cmap==q),3),2)/q_num;
    q_data=reshape(u(repmat(reshape((cmap==q),1,hx,hy),[nknm 1 1])),nknm,[])';
    %q_data=reshape(u(repmat((cmap==q),[nknm 1 1])),nknm,[])';
    mj_pure(:,q)=mean(q_data)';
    %disp(['q_data size is ' num2str(size(cov(q_data)))]);
    Cj_pure(q,:,:)=reshape(cov(q_data),1,nknm,nknm);
end
