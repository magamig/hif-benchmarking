function z=hre_pixelj_msi(x,y,fmap,mj_pure,Cj_pure,sigy)

[nq,ni]=size(fmap); % ni=6*6
nm=size(x,1)/ni;
nk=size(y,1);
Cn=sigy*sigy*eye(nk,nk);

% Compute joint statistics

H=repmat(eye(nk,nk)/ni,1,ni);
Czz=zeros(nk*ni,nk*ni);
Czx=zeros(nk*ni,nm*ni);
Cxz=zeros(nm*ni,nk*ni);
Cxx=zeros(nm*ni,nm*ni);
mz=zeros(nk*ni,1);
mx=zeros(nm*ni,1);
if nq>0
    for i=1:ni
        minx=(i-1)*nm+1;
        maxx=i*nm;
        minz=(i-1)*nk+1;
        maxz=i*nk;
        for q=1:nq
            f=fmap(q,i);
            Czz(minz:maxz,minz:maxz)=Czz(minz:maxz,minz:maxz)+f*f*reshape(Cj_pure(q,1:nk,1:nk),nk,nk);
            Cxz(minx:maxx,minz:maxz)=Cxz(minx:maxx,minz:maxz)+f*f*reshape(Cj_pure(q,nk+1:nk+nm,1:nk),nm,nk);
            Czx(minz:maxz,minx:maxx)=Czx(minz:maxz,minx:maxx)+f*f*reshape(Cj_pure(q,1:nk,nk+1:nk+nm),nk,nm);
            Cxx(minx:maxx,minx:maxx)=Cxx(minx:maxx,minx:maxx)+f*f*reshape(Cj_pure(q,nk+1:nk+nm,nk+1:nk+nm),nm,nm);
            
            mz(minz:maxz)=mz(minz:maxz)+f*mj_pure(1:nk,q);
            mx(minx:maxx)=mx(minx:maxx)+f*mj_pure(nk+1:nk+nm,q);
        end
    end
end
mzgx=mz+Czx*pinv(Cxx)*(x-mx);
Czgx=Czz-Czx*pinv(Cxx)*Cxz;

% Form estimate

z=mzgx+Czgx*H'/(H*Czgx*H'+Cn)*(y-H*mzgx);
%if rcond(Czgx)>2.2e-16
%    if inv(H'/Cn*H+inv(Czgx))>2.2e-16
%        z=(H'/Cn*H+inv(Czgx))\(H'/Cn*y-Czgx\mzgx);
%    else
%        disp('error');
%        z=pinv(H'/Cn*H+inv(Czgx))*(H'/Cn*y-Czgx\mzgx);
%    end
%else
%    disp('error');
%    if inv(H'/Cn*H+pinv(Czgx))>2.2e-16
%        z=(H'/Cn*H+pinv(Czgx))\(H'/Cn*y-pinv(Czgx)*mzgx);
%    else
%        disp('error');
%        z=pinv(H'/Cn*H+pinv(Czgx))*(H'/Cn*y-pinv(Czgx)*mzgx);
%    end
%end
z=reshape(z,nk,ni);