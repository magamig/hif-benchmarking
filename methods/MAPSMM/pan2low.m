function Out=pan2low(pan,mi,mj,mode)

[xdata,ydata]=size(pan);
hx=xdata/mi;
hy=ydata/mj;
if mode==1
    Out=zeros(hx,hy);
    for i=1:hx
        for j=1:hy
            Out(i,j)=mean(mean(pan((i-1)*mi+1:i*mi,(j-1)*mj+1:j*mj)));
        end
    end
elseif mode==2
    w=mi;
    Out=reshape(gaussian_down_sample(reshape(pan,xdata,ydata,1),w),hx,hy);
end
