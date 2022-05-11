function [ C ] = tensor_coding( D1,D2,D3,X,C1)

a=1.01;

if nargin==5
    C=C1;
else
    C=randn(size(D1,2),size(D2,2),size(D3,2));
end

notfirst=0;
    YY=ttm(tensor(C),{D1,D2,D3}, [1 2 3]);


iter=40;
res=zeros(1,iter);
res(1)=norm(double(YY(:))-X(:))+100;
for i=1:iter
    C_old=C;
    L=a*norm(D1'*D1,'fro')*norm(D2'*D2,'fro')*norm(D3'*D3,'fro');
%       L=a*norm(D1'*D1,2)*norm(D2'*D2,2)*norm(D3'*D3,2);

FC=ttm(tensor(C),{D1'*D1,D2'*D2,D3'*D3}, [1 2 3])-ttm(tensor(X),{D1',D2',D3'}, [1 2 3]);

FC=double(FC);
C=C-1.0/L*FC; 

YY=ttm(tensor(C),{D1,D2,D3}, [1 2 3]);

 res(i+1)=norm(double(YY(:))-X(:));
   if (res(i+1) / res(i))>1
     
        break
  end

  if (1/res(i+1) * res(i)) < 1.01
       
        break
  end
 
   if (res(i+1) / res(i))>1
        if notfirst == 1
            C = C_old;
     
            break
        else
            notfirst = 1;
        end
    end
  
end



