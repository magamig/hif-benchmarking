function [spectra]=nfinder_iterate(pct,filter)

nb=size(pct,1);
nr=size(pct,2);
nc=size(pct,3);
nend=nb+1;

adds=0;
E=zeros(nend,nend);
for i=1:nr
    for j=1:nc
        if adds<nend
            if filter(i,j)==1
                etest=0;
                for k=1:nend
                    if pct(:,i,j)==E(2:nend,k)
                        etest=1;
                    end
                end
                if etest==0
                    adds=adds+1;
                    E(2:end,adds)=pct(:,i,j);
                end
            end
        end
    end
end

v0=abs(det(E));

changes=1;
Etest=E;
while changes~=0
    changes=0;
    for i=1:nr
        for j=1:nc
            if filter(i,j)==1
                for k=1:nend
                    Etest(:,k)=[1; pct(:,i,j)];
                    vtest=abs(det(Etest));
                    if vtest>v0
                        E(:,k)=[1;pct(:,i,j)];
                        v0=vtest;
                        changes=changes+1;
                    else
                        Etest(:,k)=E(:,k);
                    end
                end
            end
        end
    end
end

spectra=E(2:nend,:);