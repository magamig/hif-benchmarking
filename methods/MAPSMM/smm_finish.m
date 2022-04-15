function [fmap,csum]=smm_finish(pct,prior,m,C,f)

nb=size(pct,1);
nr=size(pct,2);
nc=size(pct,3);

%disp=('');
disp('SMM Completion');

% Compute posterior class probabilities

post=postupdate2(pct,prior,m,C);
nq=size(post,1);

% Assign based on peak posterior probabolity

peak=zeros(nr,nc);
pindex=zeros(nr,nc);
for q=1:nq
    pband=reshape(post(q,:,:),nr,nc);
    I=find(pband>peak);
    pindex(I)=q;
    peak(I)=pband(I);
end

% Generate fraction planes

npure=size(f,2);
fmap=zeros(npure,nr,nc);
for i=1:nr
    for j=1:nc
        fmap(:,i,j)=f(pindex(i,j),:);
    end
end

% Generate class population vector

csum=zeros(nq,1);
for q=1:nq
    csum(q)=size(find(pindex==q),1);
end