function  A   =  sparse_tucker2( D1,D2, X,Y, lambda,A0,mu1 )
%% optimize core tensor via the ADMM

T        =    60;
 A        =    zeros( size(D1{1},2),size(D1{2},2),size(D1{3},2));
% A=A0;
V1        =    zeros( size(A) );

DTX = TensorChainProductT(X,D1,1:3);
wt=1;
N = numel(D1);
for i = ndims(X) : -1 : N+1
    wt = ones(size(X,i),1)*wt';
    wt = wt(:);
end

mu        =   0.01;

for i = 3 : -1 : 1
    [U,S] = eig(D1{i}'*D1{i});
    P{i} = U;
    wt = diag(S)*wt';
    wt = wt(:);
end
wt = reshape(wt+mu,size(A));  % diagonal vector of (ita*kron(S_1,..S_n) + 2*gamma*I)^(-1)
Z = TensorChainProductT(DTX,P,1:3);
Z = Z./wt;
bbb = TensorChainProduct(Z,P,1:3);



V2        =    zeros( size(A) );
DTX1 = TensorChainProductT(Y,D2,1:3);
N1 = numel(D2);
wt1=1;
for i = ndims(Y) : -1 : N1+1
    wt1 = ones(size(Y,i),1)*wt1';
    wt1 = wt1(:);
end
for i = 3 : -1 : 1
    [U1,S1] = eig(D2{i}'*D2{i});
    P1{i} = U1;
    wt1 = diag(S1)*wt1';
    wt1 = wt1(:);
end
wt1 = reshape(wt1+mu,size(A));  % diagonal vector of (ita*kron(S_1,..S_n) + 2*gamma*I)^(-1)
Z1 = TensorChainProductT(DTX1,P1,1:3);
Z1 = Z1./wt1;
bbb1 = TensorChainProduct(Z1,P1,1:3);



for  i  =  1:T
    Z = TensorChainProductT(mu*(A-V1),P,1:3);
Z = Z./wt;
ccc = TensorChainProduct(Z,P,1:3);
    S         =   bbb+ccc ;  
    
     Z1 = TensorChainProductT(mu*(A-V2),P1,1:3);
Z1 = Z1./wt1;
ccc1 = TensorChainProduct(Z1,P1,1:3);
    S1         =   bbb1+ccc1 ;  
    
    ddddd=(mu*(S+V1+S1+V2)+mu1*A0)/(2*mu+mu1);
 A         =  soft(ddddd, lambda/(4*mu+2*mu1));
% a11=norm(S(:)-A(:))/(norm(V1(:))+eps);
% a22=norm(S1(:)-A(:))/(norm(V2(:))+eps);
% if i>=2
% aa(i)=a11+a22
% end
    V1         =   V1 + ( S - A );
V2         =   V2 + ( S1 - A );

%     fun(i)    =   0.5*sum(sum((X-D*A).^2)) + par.lambda*sum(sum(abs(A)));
end
