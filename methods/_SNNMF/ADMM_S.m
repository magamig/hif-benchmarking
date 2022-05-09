function [S y] = ADMM_S(X,Y,F,Gt,A,S0,y0,U,D,dsf,k)
% ADMM Solver for S
B = [kron(Gt,speye(size(S0,1))); kron(speye(size(S0,2)),F*A)]; AiX = pinv(A)*X;
z = [AiX(:);Y(:)]; Btz = reshape(AiX*Gt+((F*A)')*Y,size(B,2),1);
S = S0; s = S(:); x = s; y = y0; %y = zeros(size(B,2),1);
h1 = zeros(size(B,2),1); h2 = zeros(size(B,2),1);
mu = 1e0; mufac = 1.01; mumax = 1e5; tol = 1e-5; lambda = 1e-3*k; %lambda = (3.48e-4)*k; %parameters used when dsf = 32
L0 = lgn_S(B,z,y,x,s,h1,h2,lambda,mu);
dL = 1e0; s1 = 1e0; s2 = 1e0; %convergence criteria arbitrarily initialized to 1
spacer = 12.5; %inverse frequency of how frequently to check the lagrangian, 2*spacer must be an integer
N = size(U,1); j = ones(dsf^2,1); d2 = dsf^2; rc = size(Y,2)/d2;

%Print iteration info
iter = 1; str = ['Iteration ',num2str(0), ': dL ', num2str(dL), ' s1 ', num2str(s1),...
    ' s2 ', num2str(s2)]; disp(str); tic

while abs(dL) > tol || s1 > tol || s2 > tol
    %% Update s
    t = x + (1/mu)*h1;
    s = sign(t).*max(abs(t)-lambda/mu,0);
    %% Update x
    tx = (1/2)*(s+y+(-1/mu)*(h1+h2));
    x = max(tx,0);
    %% Update y
    T = U*diag(((diag(D)+mu).^-2).*((d2*(diag(D)+mu).^-1 + d2^2).^-1))*(U'); % create T (see Supplementary notes)
    Aw_s = U*diag(((diag(D)+mu).^-1))*(U'); % create Aw_s (see "Aw" in the supplementary notes)
    r = Btz + mu*x + h2; 
    Rlong = reshape(r,N,rc*d2);
    for rcb = 1:rc
    RlongJ(:,d2*(rcb-1)+1:d2*rcb) = repmat(Rlong(:,d2*(rcb-1)+1:d2*rcb)*j,1,d2);
    end
    y = reshape(Aw_s*Rlong - T*RlongJ,rc*N*d2,1);
    %% Update h1 and h2
    h1 = h1+mu*(x-s);
    h2 = h2+mu*(x-y);
    %% Update stopping variables
    mu = min(mumax,mufac*mu); remainder = rem(iter,2*spacer);
    if remainder == 2*spacer-1
        L0 = lgn_S(B,z,y,x,s,h1,h2,lambda,mu);
    elseif remainder == 0
        L = lgn_S(B,z,y,x,s,h1,h2,lambda,mu);
        dL = (L0-L)/L0; L0 = L;
        s1 = norm(x-s,Inf);
        s2 = norm(x-y,Inf);
        avgTime = toc/(2*spacer); tic
        str = ['Iteration ',num2str(iter), ': dL ', num2str(dL), ' s1 ',... 
           num2str(s1),' s2 ', num2str(s2),'    ',num2str(avgTime),...
           ' sec/iteration']; disp(str);
    end
    iter = iter + 1;
end
S = reshape(s,size(S0,1),size(S0,2));
end

