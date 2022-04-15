function [prior,m_pure,C_pure]=ranstat(hsi,class,csum,index,p_old,m_old,C_old)

nb=size(hsi,1);
nr=size(hsi,2);
nc=size(hsi,3);
nq=size(csum,1);
npure=size(index,1);

% Update prior

prior=csum'/nr/nc;
prior=prior./sum(prior);

% Update pure class mean

m_pure=zeros(nb,npure);
for i=1:nr
    for j=1:nc
        for q=1:npure
            if class(i,j)==index(q)
                m_pure(:,q)=m_pure(:,q)+reshape(hsi(:,i,j),nb,1);
            end
        end
    end
end
for q=1:npure
    if csum(index(q))>0
        m_pure(:,q)=m_pure(:,q)/csum(index(q));
    else
        m_pure(:,q)=m_old(:,q);
        disp(['Unable to update mean: Class #' num2str(q)]);
    end
end

% Update pure class covariance

C_pure=zeros(npure,nb,nb);
for i=1:nr
    for j=1:nc
        for q=1:npure
            if class(i,j)==index(q)
                Cinc=(hsi(:,i,j)-m_pure(:,q))*(hsi(:,i,j)-m_pure(:,q))';
                C_pure(q,:,:)=reshape(C_pure(q,:,:),nb,nb)+Cinc;
            end
        end
    end
end

for q=1:npure
    if csum(index(q))>(2*nb)
        C_pure(q,:,:)=reshape(C_pure(q,:,:),nb,nb)/(csum(index(q))-1);
    else
        C_pure(q,:,:)=C_old(q,:,:);
        disp(['Unable to update covariance: Class #' num2str(q)]);
    end
end
