function [prior,m,C,f,L,J,index]=smm_reset(pct,m_pure,C_pure,L,J,npure,npure_max,nlevels)

disp('');
disp('SMM reinitialization');
disp(['Pure classes =' num2str(npure)]);
disp(['Maximum mixture classes =' num2str(npure_max)]);
disp(['Mixture levels =' num2str(nlevels)]);

% Compute fraction vector array and mixed class statistics

if nlevels==1
    f=eye(npure);
else
    f=distribute(nlevels,npure,npure_max)/nlevels;
end
nq=size(f,1);
disp(['Number of mixture classes =' num2str(nq)]);
index=zeros(npure,1);
for q=1:npure
    index(q)=nq+1-q;
end
prior=ones(1,nq)/nq;
[m,C]=mixedstat(f,m_pure,C_pure);

% Initialize SMM fit loop 

L0=loglike(pct,prior,m,C);
L=[L;L0];
J0=smm_scatter(prior,m_pure,C_pure,index);
J=[J;J0];
disp('');
disp('Iteration    L      J   ');
disp(' ------- ------- ------- ');
disp(['   0   ' num2str(L0) num2str(J0)]);