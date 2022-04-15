%% smm_init
function [prior,m,C,m_pure,C_pure,f,L,J,index]=smm_init(pct,npure,npure_max,nlevels,minit,sigma,scale,data,PC,F)

%
% [prior,m,C,m_pure,C_pure,f,L,J,index]=smm_init(pct,npure,npure_max,nlevels,minit,sigma,scale,F)
%
% Initializes the Stochastic Mixing Model (SMM) estimation process by
% determining initail
% prior probabilities (prior)
% mixed class means and  covariance matricies (m,C)
% pure class means and covariance matricies (m_pure,C_pure)
% discrete fraction vectors (f)
% initial fit matrix (L)
% scatter metric and index vector for pure classes (index)
% Initialization is based on data, usually the leading PC subspace (pct)
% number of pure classes (npure)
% maximum allowable components of a mixed class (npure_max)
% discrete mixture levels (nlevels)
% class mean initialization method (minit)
% and initial class variance.
% Mean initialization methods include: 
% (1) global mean
% (2) random pixels
% (3) span data
% (4) VCA
% (5) N-finder
% Covariance is white noise for sigma > 0 and glabal covariance (scaled by
% 'scale') otherwise.
% With NFINDER initialization, data is pre-filtered to include fraction F
% of of pixels closest to the global mean.
%
% Formats
% pct(nb,nr,nc)
% prior(1,nq)
% m(nb,nq)
% C(nq,nb,nb)
% m_pure(nb,npure)
% C_pure(npure,nb,nb)
% f(nq,npure)
% index(npure)
%

nb=size(pct,1);
nr=size(pct,2);
nc=size(pct,3);
disp('SMM initialization');
disp(['Subspace dimensionality =' num2str(nb)]);
disp(['Pure classes =' num2str(npure)]);
disp(['Maximum mixture classes =' num2str(npure_max)]);
disp(['Mixture levels =' num2str(nlevels)]);

% initialization class mean statistics

if minit==1
    disp('Global mean method');
    m_pure=zeros(nb,npure);
    m_global=mean(mean(pct,3),2); %%%
    for q=1:npure
        m_pure(:,q)=m_global;
    end
elseif minit==2
    disp('Random pixel mean method');
    m_pure=zeros(nb,npure);
    for q=1:npure
        i=ceil(rand(1)*nr);
        j=ceil(rand(1)*nc);
        m_pure(:,q)=reshape(pct(:,i,j),[],1);
    end
elseif minit==3
    disp('Space spanning mean method');
    m_pure=zeros(nb,npure);
    m_min=zeros(nb,1);
    m_max=zeros(nb,1);
    for k=1:nb
        m_min(k)=min(reshape(pct(k,:,:),nr*nc,1));
        m_max(k)=max(reshape(pct(k,:,:),nr*nc,1));
    end
    for q=1:npure
        m_pure(:,q)=m_min+(q-1)*m_max/(npure-1);
    end
elseif minit==4
    disp('VCA method');
    [ U, ~, ~ ] = vca((reshape(permute(data,[2 3 1]),[],size(data,1))/PC)', npure);
    m_pure0=(U'*PC)';
    m_pure=m_pure0(1:nb,:);
    %m_pure=U;
elseif minit==5
    if F<1
        filter=zeros(nr,nc);
        m_global=mean(mean(pct,3),2);
        C_global=cov(reshape(pct,[],nr*nc)');
        d=gdistance(pct,m_global,inv(C_global));
        dsort=sort(d(:));
        dthreshold=dsort(floor(F*nr*nc));
        filter(find(d<dthreshold))=1;
    else
        filter=ones(nr,nc);
    end
    if nb==(npure-1)
        disp(' NFINDER mean method, band extrapolated simplex');
        disp([' Percent pixels retained = ' num2str(100*F) '%']);
        m_pure=nfinder_iterate(pct,filter);
    elseif nb>(npure-1)
        disp(' NFINDER mean method, band extrapolated simplex');
        disp([' Percent pixels retained = ' num2str(100*F) '%']);
        m_pure=zeros(nb,npure);
        m_pure(1:(npure-1),:)=nfinder_iterate(pct(1:(npure-1),:,:),filter);
        mr=mean(reshape(pct(npure:nb,:,:),[],nr*nc),2)';
        for q=1:npure
            m_pure(npure:nb,q)=mr;
        end
    else
        disp(' NFINDER mean method, class extrapolated simplex');
        disp([' Percent pixels retained = ' num2str(100*F) '%']);
        m_pure=zeros(nb,npure);
        m_pure(:,1:(nb+1))=nfinder_iterate(pct,filter);
        m_global=mean(mean(pct,3),2);
        for q=(nb+2):npure
            m_pure(:,q)=m_global;
        end
    end
end

if sigma<=0
    disp(['Global covariance matrix method, scale =' num2str(scale)]);
    C_pure=zeros(npure,nb,nb);
    C_global=cov(reshape(pct,nb,[])');
    for q=1:npure
        C_pure(q,:,:)=scale*scale*C_global;
    end
else
    disp(['White noise covariance method, sigma =' num2str(sigma)]);
    C_pure=zeros(npure,nb,nb);
    for q=1:npure
        for i=1:nb
            C_pure(q,i,i)=sigma^2;
        end
    end
end

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

L=[];
J=[];
L0=loglike(pct,prior,m,C);
J0=smm_scatter(prior,m_pure,C_pure,index);
L=[L;L0];
J=[J;J0];
disp('');
disp('Iteration    L      J   ');
disp(' ------- ------- ------- ');
disp(['   0   ' num2str(L0) num2str(J0)]);


