function hsic = MAPSMM_fusion(HSI,MSI)
%--------------------------------------------------------------------------
% USAGE
%       hsic = MAPSMM_fusion(HSI,MSI)
%
% INPUT
%       HSI : Low-spatial-resolution HS image
%       MSI : MS image 
%
% OUTPUT
%       hsic: High-spatial-resolution HS image 
%
% Reference:
% RESOLUTION ENHANCEMENT OF HYPERSPECTRAL IMAGERY USING MAXIMUM A
% POSTERIORI ESTIMATION WITH A STOCHASTIC MIXING MODEL
% PhD thesis
% Michael Theodore Eismann
% University of Dayton (2004)
%
%--------------------------------------------------------------------------

[hx,hy,~] = size(HSI);
[xdata,~,~] = size(MSI);
w = xdata/hx;

HSI = reshape(HSI,hx*hy,[]);
PC = pca(HSI);
HSI = permute(reshape(HSI,hx,hy,[]),[3 1 2]);
HSI_pc = permute(reshape(PC.pca,hx,hy,[]),[3 1 2]);

%% SMM
npure=4;
smm = smm_run(HSI_pc,PC.a,npure);
%%% modified
[~,I] = max(smm.fmap,[],1);
if numel(unique(I)) < npure
    smm = smm_run(HSI_pc,PC.a,npure+1);
    [~,I] = max(smm.fmap,[],1);
    uniqueidx = unique(I);
    smm.C_pure = smm.C_pure(uniqueidx,:,:);
    smm.m_pure = smm.m_pure(:,uniqueidx);
    smm.fmap = smm.fmap(uniqueidx,:,:); % (npure,nr,nc)
end

%% Hyperspectral resolution enhancement
method = 15;
nkt = npure+1; % Demensionality of leading subspace 5
%snr = 100;
mi = w;
mj = w;
nk = size(HSI_pc,1); % Number of spectral channels of HSI
sigma = sqrt(PC.eig(nkt+1,1));
sigJ = sqrt(PC.eig(1,1));
V = PC.a; % Transform matrix of PCA
ilow = size(HSI,3); % hyper x
jlow = size(HSI,2); % hyper y
ni = ilow*mi; % h-m x
nj = jlow*mj; % h-m y

% Read multi
% msi(nm,nj,n)
MSI = permute(MSI,[3 1 2]);
nm = size(MSI,1); % Number of spectral channels of MSI

% Compute noise parameter
%sigx = std(MSI(:))/snr;
%MSI = MSI+sigx*randn(nm,nj,ni);
%disp(['Multispectral SNR =' num2str(std(MSI(:))/sigx)]);
sigy = sigma/sqrt(mi*mj);

%disp(['sigma =' num2str(sigma)]);
%disp(['sigy =' num2str(sigy)]);
%disp(['sigJ =' num2str(sigJ)]);

Cn = sigy^2*eye(nk,nk);
Cn = Cn(1:nkt,1:nkt);
disp(['First PC SNR =' num2str(sigJ/sigy)]);

% Read in high resolution SMM statistics and compute low resolution map
if method>6
    %disp('Reading SMM data');
    nq=size(smm.fmap,1);
    C_pure=smm.C_pure(:,1:nkt,1:nkt);
    m_pure=smm.m_pure(1:nkt,:);
    fmaplow=smm.fmap; % (npure,nr,nc)
    clear smm;
end

% Perform resolution enhancement
disp('Performing resolution enhancement');
pcie = zeros(nk,nj,ni);
pcie(1:nkt,:,:) = hre_joint_msi(MSI,HSI_pc(1:nkt,:,:),fmaplow,m_pure,C_pure,sigy);
pcie(nkt+1:nk,:,:) = hre_replicate2(HSI_pc(nkt+1:nk,:,:),mj,mi);
disp('Performing PC transformation');
pcie = reshape(permute(pcie,[2 3 1]),[],nk);
hsic = pcie/V;
hsic = hsic+repmat(PC.mean,size(hsic,1),1);
hsic = reshape(hsic,nj,ni,[]);