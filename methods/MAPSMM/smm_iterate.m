%function [prior,m,C,m_pure,C_pure,L,J,csum]=smm_iterate(pct,prior,m,C,m_pure,C_pure,L,J,f,index,niter,r)
function [prior,m,C,m_pure,C_pure,fmap,L,J]=smm_iterate(pct,prior,m,C,m_pure,C_pure,L,J,f,index,niter)

% Perform SMM iterations
nq=size(m,2);
npure=size(m_pure,2);
for iter=1:niter
    Out=postupdate(pct,prior,m,C,f);
    %[class,csum]=ranmix(post);
    %[prior2,m_pure,C_pure]=ranstat(pct,class,csum,index,prior,m_pure,C_pure);
    %[m,C]=mixedstat(f,m_pure,C_pure);
    m=Out.m;
    C=Out.C;
    prior=Out.prior;
    fmap=Out.fmap;
    m_pure=m(:,nq-npure+1:nq);
    C_pure=C(nq-npure+1:nq,:,:);
    %for q=1:nq
    %    rank(reshape(C(q,:,:),size(C,2),[]))
    %end
    %break;
    L0=loglike(pct,prior,m,C);
    J0=smm_scatter(prior,m_pure,C_pure,index);
    L=[L;L0];
    J=[J;J0];
    disp(['    ' num2str(iter) '    ' num2str(L0) '    ' num2str(J0)]);
    %if r~=1
    %    prior=prior2;
    %end
end

%if r==1
%    prior=prior2;
%end