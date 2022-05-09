function [A] = ADMM_A(S,G,F,X,Y,A0)
% ADMM Solver for A
SG = S*G; C = [kron((SG)',eye(size(F,2))); kron(S',F)]; f = [X(:); Y(:)];
Ctf2 = 2*reshape(X*(SG)'+(F')*(Y*S'),size(C,2),1);
cols = size(C,2); T0 = kron(SG*(2*SG'),eye(size(X,1))) + kron(S*(S'),(2*F')*F);
A = A0; rho = A(:); alpha = rho; y = zeros(cols,1);
mu = 1e2; mumax = 1e7; tol = 1e-5; %parameters used when dsf = 32
L0 = lgn_A(C,y,f,rho,alpha,mu); dL = 10.0; s1 = 10.0;
spacer = 12.5; %inverse frequency of how frequently to check the lagrangian, 2*spacer must be an integer

%Print iteration info
iter = 1; str = ['Iteration ',num2str(0), ': dL ', num2str(dL), ' s1 ',...
    num2str(s1)]; disp(str); tic;

while abs(dL) > tol || (s1 > tol)
    %% Update alpha
    T = T0 + mu*eye(size(T0,1));
    alpha = T\(Ctf2 + mu*rho + y); 
    %% Update rho
    rho = max(alpha - (1/mu)*y,0);
    %% Update y
    y = y + mu*(rho - alpha); 
    %% Update mu and the stopping variables
    mu = min(mumax,1.01*mu);  remainder = rem(iter,2*spacer);
    if remainder == 2*spacer-1
        L0 = lgn_A(C,y,f,rho,alpha,mu);       
    elseif remainder == 0
        L = lgn_A(C,y,f,rho,alpha,mu);
        dL = (L0-L)/L0; L0 = L;
        s1 = norm(rho - alpha,Inf);
        avgTime = toc/(2*spacer); tic
        str = ['Iteration ',num2str(iter), ': dL ', num2str(dL), ' s1 ',...
            num2str(s1),'    ',num2str(avgTime),' sec/iteration']; disp(str); 
    end
    iter = iter + 1;
end
for col = 1:size(A0,2)
   for row = 1:size(A0,1)
       A(row,col) = rho(row + size(A0,1)*(col - 1));
   end
end
end