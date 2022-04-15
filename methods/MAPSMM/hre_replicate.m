function hsie=hre_replicate(hsi,mj,mi)

%Initialization

[nk,njl,nil]=size(hsi);
nj=njl*mj;
ni=nil*mi;
hsie=zeros(nk,nj,ni);

% Replicate band-by-band

for k=1:nk
    %bandlow=reshape(hsi(k,:,:),njl,nil);
    for i=1:nil
        imin=(i-1)*mi+1;
        imax=i*mi;
        for j=1:njl
            jmin=(j-1)*mj+1;
            jmax=j*mj;
            hsie(k,jmin:jmax,imin:imax)=hsi(k,j,i);
        end
    end
end
