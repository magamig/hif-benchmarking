function L=loglike(hsi,prior,m,C)

nb=size(hsi,1);
nr=size(hsi,2);
nc=size(hsi,3);
nq=size(prior,2);

psum=zeros(nr,nc);

for q=1:nq
    p=gausspdf(hsi,m(:,q),reshape(C(q,:,:),nb,nb));
    psum=psum+prior(1,q)*p;
end

I=find(psum~=0);
%disp(num2str(size(I,1)));
%sum(log(psum(I)))
L=sum(log(psum(I)))/size(I,1);