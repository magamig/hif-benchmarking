function [ W, b] = learn_spectral_similarties(HSI3,eta)

% D=HSI3*HSI3'/(HSI3*HSI3'+eta);
% W=-D/diag(diag(D))

B=HSI3';
for i = 1:size(HSI3)-1
        tar_tmp =HSI3(i,:);
        ref_tmp = HSI3(i+1,:);
        cc = corrcoef(tar_tmp(:),ref_tmp(:));
        out(1,i) = cc(1,2);
end

b(1)=0;
    j=2;
for i = 1:size(HSI3)-3
       if (out(i+1)<out(i))&&(out(i+1)<out(i+2))&&(out(i+1)<0.97)
           b(j)=i+1;
           j=j+1;
       end
end
b(j)=size(HSI3,1);
for i=1:length(b)-1
    A=B(:,b(i)+1:b(i+1));
 W{i}= (A'*A+eta*eye(size(A,2)))\(A'*A);  
 
%    W1=-D/diag(diag(D));
%  W{i}=W1-diag(diag(W1)); 
end


% B=HSI3';

% D=(B'*B+eta)\(B'*B);
% W=-D/diag(diag(D));
% W=W-diag(diag(W));


%  W=(B'*B+eta*eye(size(B,2)))\(B'*B);