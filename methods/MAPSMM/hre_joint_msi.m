function hsie=hre_joint_msi(msi,hsi,fmap,m_pure,C_pure,sigy)

% Initialization

[nm,nj,ni]=size(msi);
[nk,njl,nil]=size(hsi);
nq=size(fmap,1);
mj=floor(nj/njl);
mi=floor(ni/nil);
hsie=zeros(nk,njl*mj,nil*mi);
f=zeros(nq,mi*mj);
z=zeros(nk,mi*mj);

% Perform bilinear interpolation of fraction planes

j1=floor(mj/2)+1;
j2=nj-floor(mj/2);
i1=floor(mi/2)+1;
i2=ni-floor(mi/2);
[jhi,ihi]=meshgrid(i1:1:i2,j1:1:j2);
[jlow,ilow]=meshgrid(((mi+1)/2):mi:(ni-(mi-1)/2),((mj+1)/2:mj:(nj-(mj-1)/2)));

fmapl=fmap;
fmap=zeros(nq,njl*mj,nil*mi);
% % % for q=1:nq
% % %     bandlow=reshape(fmapl(q,:,:),njl,nil);
% % %     for i=1:nil
% % %         imin=(i-1)*mi+1;
% % %         imax=i*mi;
% % %         for j=1:njl
% % %             jmin=(j-1)*mj+1;
% % %             jmax=j*mj;
% % %             fmap(q,jmin:jmax,imin:imax)=fmapl(q,j,i);
% % %         end
% % %     end
% % %     fmap(q,j1:j2,i1:i2)=interp2(jlow,ilow,bandlow,jhi,ihi,'bilinear');
% % %     %%%figure(3);
% % %     %%%subplot(3,3,q); imagesc(reshape(fmap(q,:,:),nj,ni)'); axis image
% % % end

for q=1:nq
    fmap(q,:,:) = imresize(reshape(fmapl(q,:,:),njl,nil),mj);
end

% Compute joint statistics for pure classes

u=zeros(nk+nm,njl,nil);
u(1:nk,:,:)=hsi;
for k=1:nm
    u(nk+k,:,:)=reshape(pan2low(reshape(msi(k,:,:),nj,ni),mj,mi,2),1,njl,nil);
end
%%%for k=1:nk+nm
%%%    figure(18); subplot(3,4,k); imagesc(reshape(u(k,:,:),njl,nil));
%%%end
%cmap=zeros(njl,nil);
%for q=1:nq
%    cmap(fmapl(q,:,:)==1)=q;
%end
[~,I]=max(fmapl,[],1);
cmap=reshape(I,njl,nil);
%%%figure(19); imagesc(reshape(sum(u,1),njl,nil));
[mj_pure,Cj_pure]=class_stats(u,cmap); % ??????
% figure(10); subplot(1,2,1); imagesc(reshape(sum(Cj_pure,1),nk+nm,[]));
% subplot(1,2,2); plot(mj_pure);

% Perform MAP estimation based on SMM statistics

for i=1:nil
    imin=(i-1)*mi+1;
    imax=i*mi;
    for j=1:njl
        jmin=(j-1)*mj+1;
        jmax=j*mj;
        x=reshape(msi(:,jmin:jmax,imin:imax),nm*mi*mj,1);
        y=reshape(hsi(:,j,i),nk,1);
        for q=1:nq
            f(q,:)=reshape(fmap(q,jmin:jmax,imin:imax),1,mi*mj);
        end
        z=hre_pixelj_msi(x,y,f,mj_pure,Cj_pure,sigy);
        for k=1:nk
            hsie(k,jmin:jmax,imin:imax)=reshape(z(k,:),mj,mi);
        end
    end
end

        