%% hre_run
% Hyperspectral resolution enhancement with multispectral data using SMM
% 
method=15;
%variant=1;
%alpha=1;
%niter=20;
mi=4; % Pixel degradation in x
mj=4; % Pixel degradation in y
nkt=10; % Demensionality of leading subspace
snr=100;
%norm=0;
%reshpnse=0;
%truth=1;
%filter=2;
%fname_smm

% Read in low resolution hyperspectral data

hsilow=
pcilow=
nk=size(data.hsi,1); % Hyperバンド数
sigma=sqrt(data.values(nkt+1)); % PCA掛けたときのλ(nkt+1番目)
sigJ=sqrt(data.values(1)); % PCA掛けたときのλ(1番目)
V=data.V; % PCAの変換行列
ilow=size(hsilow,3); % hyper x
jlow=size(hsilow,2); % hyper y
ni=ilow*mi; % h-m x
nj=jlow*mj; % h-m y

% Read multi
% msi(nm,nj,n)
nm=size(msi,1); % Multiバンド数

% Compute noise parameter

sigx=std(msi(:)/snr);
msi=msi+sigx*randn(nm,nj,ni);
disp(['Multispectral SNR =' num2str(std(msi(:))/sigx)]);
sigy=sigma/sqrt(mi*mj);
Cn=sigy^2*eye(nk,nk);
Cn=Cn(1:nkt,1:nkt);
disp(['First PC SNR =' num2str(sig1/sigy)]);

% Read in high resolution SMM statistics and compute low resolution map

if method>6
    disp('Reading SMM data');
    smm=open(fname_smm);
    nq=size(smm.fmap,1);
    C_pure=smm.C_pure(:,1:nkt,1:nkt);
    m_pure=smm.m_pure(1:nkt,:);
    fmaplow=smm.fmap;
    clear smm;
end

% Perform resolution enhancement

disp('Performing resolution enhancement');
pcie=zeros(nk,nj,ni);
pcie(1:nkt,:,:)=hre_joint_msi(msi,pcilow(1:nkt,:,:),fmaplow,m_pure,C_pure,sigy);
pcie(nkt+1:nk,:,:)=hre_replicate(pcilow(nkt+1:nk,:,:),mj,mi);
disp('Performing PC transformation');
hsic=pereverse(V,pcie);

