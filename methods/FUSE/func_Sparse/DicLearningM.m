%% train dictionary by solving the problem
%
%   min_{D,alpha} ||x-D*alpha||_2^2+lamda*||alpha||_1
%

%%                                    input
% X samples organized in column to train the dictionary

%
%%%%%%%%%%%%%%%%%%%%%%%%input parameters for dictionary learning%%%%%%%%%%%
%  param.K                    number of atoms in the trained dictionary
%                             defaul: 256
%  param.T                    number of iterations in the trained dictionary
%                             defaul: 500
%  param.patchnum             number of patches use in one ieration in the trained dictionary
%                             defaul: 64
%  param.lamda                parameter in SUNSAL to solve the 
%                             defaul: 1.8
%  param.err                  tolerant error in the SUNSAL algorithm
%                             defaul: 1e-2

%%%%%%%%%%%%%%%%%%%%%%%output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  D                         trained dictionary 
%  A, B                      information record in the iteration of training dictionary.


function [D,A,B] = DicLearningM(X,param)

if isfield(param, 'K')
    k = param.K;
else
    k = 256; 
end
if isfield(param, 'T')
    T = param.T;
else
    T = 500; 
end
if isfield(param, 'patchnum')
    patchnum = param.patchnum;
else
    patchnum = 64; 
end
if isfield(param, 'lamda')
    lambda = param.lamda;
else
    lambda = 0.11; 
end
if isfield(param, 'err')
    err = param.err;
else
    err = 1e-3; 
end


t0 = 1e-3;
m = size(X,1);
A = zeros(k,k);
B = zeros(m,k);
A = t0*eye(k);


DCT=zeros(m,k);
for ii=0:1:k-1,
    V=cos([0:1:m-1]'*ii*pi/k);
    if ii>0, V=V-mean(V); end;
    DCT(:,ii+1)=V/norm(V);
end;
D = DCT;
D = D./repmat(sqrt(sum(abs(D).^2)),[size(D,1) 1]);
% figure(210);displayDic(D);title('Initial Dictionary');drawnow
B = t0*D;

A = sparse(A);
B = sparse(B);

X = X./repmat(sqrt(sum(X.*conj(X))),size(X,1),1);

% S=2055615866; randn('seed', S);
% D=randn(m,k)+sqrt(-1)*randn(m,k);
% D=D./repmat(sqrt(sum(abs(D).^2)),[size(D,1) 1]);

% rng('default');
% Sele = randi(size(X,2),1,param.T);
% patchnum = param.patchnum;
Sele = randi(size(X,2),1,T*patchnum);
% Alpha_all = zeros(k,size(X,2));
% energy = zeros(T,1);

for ite = 1:T
    %% Randomly select the patches to train the dictionary 
    index = Sele(1,(ite-1)*patchnum+1:ite*patchnum);
%     index = ones(1,patchsize);
    Xi = X(:,index);
    
%     indeend = min(Sele(1,ite)+patchsize,size(X,2));
%     Xi = X(:,Sele(1,ite):indeend);

%     Xi = X(:,Sele(1,ite));
    
    %creat initial dictionary
    
%     alpha0 = zeros(k,size(Xi,2));    
%     alphat=mexL1L2BCD(Xi,D,alpha0,listgroups,parambcd);

%     lambda = param.lamda;
%     alphat = sunsal(D,Xi,'lambda',lambda,'verbose','yes');
%     alphat = sunsal(D,Xi,'lambda',lambda);

    alphat = sunsal_simple(D,Xi,lambda,err);
% for tttt = 1:patchnum
%     ttempalphat = lars(D,Xi(:,tttt),'lasso');
%     alphat(:,tttt) = ttempalphat(:,size(ttempalphat,2));
% end
%     [alphat, Xvar, paramm] = EMBGAMP(Xi, D) ;
    alphat(abs(alphat)<1e-4) = 0;
    alphat = sparse(alphat);
%     energy(ite) = norm(D*alphat-Xi,'fro')^2+lambda*norm(alphat,1);
    
%     delta=sqrt(kais)*param.sigma;
%     q = 1;
%     [alphat,res1,res2,res3] = csunsal_p(D,Xi,'Q',q,'delta',delta,'ADDONE','no',...
%         'POSITIVITY','no','AL_iters',100,'verbose','no');
%     alphat(abs(alphat)<1e-4) = 0;
    %% Online Dictionary Learning      
    [D,A,B]= ODicL(alphat,Xi,D,A,B,ite);
%     fprintf('Iteration: %d\n',ite);
end
D=D./repmat(sqrt(sum(abs(D).^2)),[size(D,1) 1]);