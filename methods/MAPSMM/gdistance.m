function arg=gdistance(hsi,m,C)

nb=size(hsi,1);
nr=size(hsi,2);
nc=size(hsi,3);

arg=zeros(nr,nc);
for i=1:nr
    for j=1:nc
        spectrum=reshape(hsi(:,i,j),nb,1);
        %arg(i,j)=real(0.5*(spectrum-m)'/C*(spectrum-m));
        if rcond(C)>2.2e-16
            arg(i,j)=real(0.5*(spectrum-m)'/C*(spectrum-m));
        else
        %    disp('');
        %    disp('Error')
        %    arg(i,j)=real(0.5*(spectrum-m)'*pinv(C)*(spectrum-m));
            arg(i,j)=1e100;
        end
    end
end
