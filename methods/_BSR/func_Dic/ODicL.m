function [D,A,B]= ODicL(alphat,Xi,D,A,B,ite)
%     for tempindex = 1:patchsize
%         A = A+alphat(:,tempindex)*alphat(:,tempindex)';
%         B = B+Xi(:,tempindex)*alphat(:,tempindex)';
%         for j = 1:k
%             if A(j,j)~=0
%                 uj = (B(:,j)-D*A(:,j))/A(j,j)+D(:,j);
%                 dj = uj/norm(uj);
%                 D(:,j) = dj;
%             end
%         end
%     end    
    
    %%
%     if ite<patchnum
%         theta = ite*patchnum;
%     else
%         theta = patchnum*patchnum+ite-patchnum;
%     end
%     beta = (theta+1-patchnum)/(theta+1);

[m,patchnum]=size(Xi);     % The size of patches
k=size(alphat,1);          % The number of atoms

rho = 2;
beta = (1-1/ite)^rho;

Atemp = sparse(zeros(k,k));
Btemp = sparse(zeros(m,k));
for tt = 1:patchnum
    Atemp = Atemp+alphat(:,tt)*alphat(:,tt)';
    Btemp = Btemp+Xi(:,tt)*alphat(:,tt)';
end
A = beta*A+Atemp;
B = beta*B+Btemp;
        
%     A = A+alphat*alphat'-Alpha_all(:,Sele(1,ite))*Alpha_all(:,Sele(1,ite))';
%     B = B+Xi*alphat'-Xi*Alpha_all(:,Sele(1,ite))';    
%     Alpha_all(:,Sele(1,ite)) = alphat;
    
for j = 1:k
    if A(j,j)~=0
        uj = (B(:,j)-D*A(:,j))/A(j,j)+D(:,j);
        dj = uj/max(norm(uj),1);
        D(:,j) = dj;
    end
end
D = abs(D);
%     for ttt = 1:5
%         for j = 1:k
%             if A(j,j)~=0
%                 uj = (B(:,j)-D*A(:,j))/A(j,j)+D(:,j);
%                 dj = uj/max(norm(uj),1);
%                 D(:,j) = dj;
%             end
%         end
%     end
%     for j = 1:k        
%         D(:,j) = D(:,j)/norm(D(:,j));        
%     end
