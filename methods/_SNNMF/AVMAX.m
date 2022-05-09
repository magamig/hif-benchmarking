function [A_est, time, iter_cnt, index] = AVMAX(XX, N)

%=====================================================================
% Programmers: 
% Tsung-Han Chan, E-mail: thchan@ieee.org  
% A. ArulMurugan, E-mail: aareul@ieee.org
% Date: Sept, 2010
%======================================================================
% A implementation of AVMAX
% [A_est,sn_est, time, iter_cnt] = AVMAX(X,N,show_flag)
%======================================================================
%  Input
%  X is M-by-L data matrix where M is the spectral bands (or observations) and L is the number of pixels (data length).   
%  N is the number of endmembers (or sources).
%  show_flag: 1- display current information in AVMAX, and 0 - otherwise 
%----------------------------------------------------------------------
%  Output
%  A_est is M-by-N: estimated endmember signatures (or mixing matrix) obtained by AVMAX.
%  time is the computation time (in secs). 
%  iter_cnt is the passed number of iterations in AVMAX. 
%  index is the set of indices of the pure pixels identified by AVMAX
%========================================================================


t0 = clock;
%----------- Define default parameters------------------
TOL_obj = 1e-6;    % convergence tolerance
val = norm(XX,'fro')^2/size(XX,2);
X = XX(:,sum(XX.^2)>0.01*val);
[M,L] = size(X);
Xn = X./(ones(M,1)*sum(X));
d = mean(Xn,2); 
U = Xn-d*ones(1,L); 
OPTS.disp = 0;
[C D] = eigs(U*U',N-1,'LM',OPTS); 
Xd = C'*U; % dimension reduced data
%--------Step 1 and Step 2: a feasible initialization-------------
r = 1;
while r<N
    index=random('unid',L,1,N); 
    E = Xd(:,index);
    E =[E;ones(1,N)];
    r = rank(E);
end
obj0 = abs(det(E))/factorial(N-1);
%----------Step 3 and Step 4.--------------
index = zeros(1,N);
 j = 0; rec = 1; iter_cnt = 0;
 while (rec > TOL_obj) & (iter_cnt<10*N)
     j = j+1;
     if j>N; j=1; end 
    b=[];
    for i=1:N;
        Eij=[E(1:i-1,1:j-1),E(1:i-1,j+1:N);E(i+1:N,1:j-1),E(i+1:N,j+1:N)];  % (N*(N-1)^eta)*N columns
        b=[b;(-1)^(i+j)*det(Eij)];  
    end
    b = b(1:N-1);
    b([b~=0]) = b([b~=0])./(max(abs(b([b~=0])))*ones(length(find(b~=0)),1)); 
    [val ind]=max(Xd'*b); 
    index(j) = ind;
    E(1:N-1,j) = Xd(:,ind);
   
    %------Step 5 and Step 6---------------------
    if j == N
        iter_cnt = iter_cnt+1;
        rec = abs(obj0-abs(det(E))/factorial(N-1))/obj0;
        obj0 = abs(det(E))/factorial(N-1);
%         if show_flag
%             disp(' ');
%             disp(strcat('Number of iterations: ', num2str(iter_cnt)))
%             disp(strcat('Relative change in objective: ', num2str(rec)))
%             disp(strcat('Current volume: ', num2str(obj0)))
%         end
    end
 end
%--------Output------------------
A_est = X(:,index); %C*E(1:N-1,:)+d*ones(1,N);
time = etime(clock,t0);
