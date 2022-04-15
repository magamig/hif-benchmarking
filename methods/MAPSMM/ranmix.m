function [class,csum]=ranmix(post)

nq=size(post,1);
nr=size(post,2);
nc=size(post,3);

pmin=zeros(nr,nc);
rand('state',sum(100*clock));
prob=rand(nr,nc);
class=zeros(nr,nc);
csum=zeros(nq,1);

pmax=pmin+reshape(post(1,:,:),nr,nc);
I=find(prob<=pmax);
class(I)=1;
csum(1,1)=size(I,1);
pmin=pmax;

if nq>1
    for q=2:nq
        pmax=pmin+reshape(post(q,:,:),nr,nc);
        I=logical((prob>pmin).*(prob<=pmax));
        class(I)=q;
        csum(q,1)=size(I,1);
        pmin=pmax;
    end
end
