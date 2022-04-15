function hsie=hre_replicate2(hsi,mj,mi)

%Initialization

[nk,njl,nil]=size(hsi);
nj=njl*mj;
ni=nil*mi;
hsie=zeros(nk,nj,ni);

% interpolation

for b = 1:nk
    hsie(b,:,:) = imresize(reshape(hsi(b,:,:),njl,nil),[nj,ni]);
end