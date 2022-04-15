function J=smm_scatter(prior,m_pure,C_pure,index)

nq=size(prior,2);
[nb,npure]=size(m_pure);

psum=0;
p=zeros(1,npure);

for q=1:npure
    p(q)=prior(index(q));
    psum=psum+p(q);
end
p=p/psum;

Sw=zeros(nb,nb);
m0=zeros(nb,1);
for q=1:npure
    Sw=Sw+p(q)*reshape(C_pure(q,:,:),nb,nb);
    m0=m0+p(q)*m_pure(:,q);
end

Sb=zeros(nb,nb);
for q=1:npure
    m=m_pure(:,q)-m0;
    Sb=Sb+p(q)*(m*m');
end

J=real(trace(pinv(Sw)*Sb));