function [ B ] = tensorsparsecoding( D1,D2,D3,X,phi,HSI)
t0=1;
a=1;
fa=2e-04;
C=randn(size(D1,2),size(D2,2),size(D3,2));
% C=tucker_als(tensor(X),[size(D1,2),size(D2,2),size(D3,2)]);
B0=C;
if size(D3,2)>1
    YY=ttm(tensor(B0),{D1,D2,D3}, [1 2 3]);

else
    d1=D1*B0;
    d2=D2*d1';
    d3=d2';
    d4=reshape(d3,[1 size(D1,1)*size(D2,1)]);
    d5=D3*d4;
    YY=Fold(d5,[ 8 8 3],3);
end
iter=100;
mse=zeros(1,iter);
mse(1)=norm(double(YY(:))-X(:));
for i=1:iter
   % L=a*norm(D1'*D1,'fro')*norm(D2'*D2,'fro')*norm(D3'*D3,'fro');
      L=a*norm(D1'*D1,2)*norm(D2'*D2,2)*norm(D3'*D3,2);
if size(D3,2)==1
    FC=double(D3'*D3*ttm(tensor(C),{D1'*D1,D2'*D2}, [1 2 ]))-double(ttm(tensor(X),{D1',D2',D3'}, [1 2 3]));
else
FC=ttm(tensor(C),{D1'*D1,D2'*D2,D3'*D3}, [1 2 3])-ttm(tensor(X),{D1',D2',D3'}, [1 2 3]);
end
FC=double(FC);
H=C-1.0/L*FC; 
%  B1=sign(double(H)).*(max(abs(H)-fa/L,0));
B1=max(H-fa/L,0);
t1=(1+sqrt(1+4*t0^2))/2;
C=B1+(t0-1)/t1*(B1-B0);
t0=t1;
if size(D3,2)>1
    YY=ttm(tensor(B1),{D1,D2,D3}, [1 2 3]);

else
    d1=D1*B1;
    d2=D2*d1';
    d3=d2';
    d4=reshape(d3,[1 size(D1,1)*size(D2,1)]);
    d5=D3*d4;
    YY=Fold(d5,[ 8 8 3],3);
end


% mse(i+1)=1/2*norm(double(YY(:))-X(:))+fa*size(find(B1(:)),1);
 mse(i+1)=norm(double(YY(:))-X(:));
%  if size(D3,2)>1
%     YY1=ttm(tensor(B1),{D1,D2,phi}, [1 2 3]);
% 
% else
%     d1=D1*B1;
%     d2=D2*d1';
%     d3=d2';
%     d4=reshape(d3,[1 size(D1,1)*size(D2,1)]);
%     d5=phi*d4;
%     YY1=Fold(d5,[ 8 8 31],3);
% end
% 
%  
%  YY1=double(im2uint8(double(YY1)));
%  HSI1=double(im2uint8(double(HSI)));
% 
% mse2(i)=getrmse(YY1,HSI1);
if abs(mse(i+1)-mse(i))/mse(i)<=0.02
    break
else
     B0=B1;
    
 end

end

B=double(B0);
