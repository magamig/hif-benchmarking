function post=postupdate2(hsi,prior,m,C)

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

post=zeros(nq,nr,nc);

for q=1:nq
    if prior(1,q)>0
        post(q,:,:)=prior(1,q)*gausspdf(hsi,m(:,q),reshape(C(q,:,:),nb,nb));
    end    
end
post(post<0)=0;

psum=reshape(sum(post,1),nr,nc);

I=find(psum==0);

for q=1:nq
    pq=reshape(post(q,:,:),nr,nc);
    ps=psum;
    pq(I)=1/nq;
    ps(I)=1;
    post(q,:,:)=pq./ps;
end