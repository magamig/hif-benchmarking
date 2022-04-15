function [m,C]=mixedstat(f,m_pure,C_pure)

nq=size(f,1);
npure=size(f,2);
nb=size(m_pure,1);

C=zeros(nq,nb,nb);
m=zeros(nb,nq);

for q=1:nq
    for c=1:npure
        C(q,:,:)=C(q,:,:)+(f(q,c)^2)*C_pure(c,:,:);
        m(:,q)=m(:,q)+f(q,c)*m_pure(:,c);
    end
end
