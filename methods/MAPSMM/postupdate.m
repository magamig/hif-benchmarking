function Out=postupdate(hsi,prior,m,C,f)

% hsi (nb,nr,nc)
% prior (1,nq)
% m (nb,nq)
% C (nq,nb,nb)
%
%

nb=size(hsi,1);
nr=size(hsi,2);
nc=size(hsi,3);
nq=size(prior,2);
npure=size(f,2);

post0=zeros(nq,nr,nc);
post=zeros(nq,nr,nc);
fmap=zeros(npure,nr,nc);
%for q=1:nq
%    if prior(1,q)>0
%        post(q,:,:)=prior(1,q)*gausspdf(hsi,m(:,q),reshape(C(q,:,:),nb,nb));
%    end    
%end
for q=1:nq
     post0(q,:,:)=gausspdf(hsi,m(:,q),reshape(C(q,:,:),nb,nb))/sum(sum(gausspdf(hsi,m(:,q),reshape(C(q,:,:),nb,nb)))); 
end 

for r=1:nr
    for c=1:nc
        for q=1:nq
            post(q,r,c)=post0(q,r,c)*prior(1,q)/sum(post0(:,r,c)); 
        end
        fmap(:,r,c)=reshape(sum(f.*repmat(reshape(post(:,r,c),[],1),1,npure))/sum(sum(f.*repmat(reshape(post(:,r,c),[],1),1,npure))),[],1,1);
    end
end

for q=1:nq
    prior(1,q)=mean(reshape(post(q,:,:),[],1));
    m(:,q)=sum(sum(hsi.*repmat(post(q,:,:),[nb 1 1]),3),2)/sum(reshape(post(q,:,:),[],1));
    hsi2=reshape(hsi,nb,[]);
    post3=repmat(reshape(post(q,:,:),1,[]),nb,1);
    C(q,:,:)=(hsi2.*post3)*hsi2'/sum(reshape(post(q,:,:),[],1));
end

Out.post=post;
Out.prior=prior;
Out.m=m;
Out.C=C;
Out.fmap=fmap;






psum=reshape(sum(post,1),nr,nc);

I=find(psum==0);

for q=1:nq
    pq=reshape(post(q,:,:),nr,nc);
    ps=psum;
    pq(I)=1/nq;
    ps(I)=1;
    post(q,:,:)=pq./ps;
end
