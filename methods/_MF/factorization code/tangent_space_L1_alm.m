function [deltaA,deltaX,lambdaHat,totalIt,relRes] = tangent_space_L1_alm(A,X,eta,lambda,clampType)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% tangent_space_L1_min_interior_point
%
%   Solves the optimization problem
%
%     minimize     ||X + deltaX||_1
%     subject to   A deltaX + deltaA X = 0,  
%                  < A_i, deltaA_i >   = 0,  i = 1...n
%                  ||deltaA||_F^2 + ||deltaX||_F^2 <= eta^2
%
%   Inputs:
%     A - m x n, presumably an estimate of a sparsifying basis
%     X - n x p, presumably an estimate of sparse coefficients wrt the basis
%     eta - scalar bound on the step length
%
%   Outputs:
%     deltaA -- step direction in A
%     deltaX -- step direction in X
%
%   Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

[m,n] = size(A);
[n,p] = size(X);
mn = m*n; mp = m*p; np = n*p;

if m == n,
    B_info     = P_B_precomputations_square_new(A,X);
    P_B_handle = @(x) P_B_square(x,B_info);
else
    B_info     = P_B_precomputations_new(A,X);
    P_B_handle = @(x) P_B_new(x,B_info);
end

xCur = [ zeros(mn,1); vec(X) ];
w    = [ zeros(mn,1); ones(np,1) ];
r    = P_B_handle(xCur);

% Solve an L1 minimization problem over x = [vec(dA) vec(X+dX)], with 
%  w = [ 0 ... 0 1 ... 1 ]'.
%
%  minimize     w' * |x|  
%  subject to   P_B x = P_B [0 vec(X)]
%               || x - [0 vec(X)] ||_2 <= eta,
%
%  where  B = [ X' tensor I  I tensor A ]
%             [ C_A'         0          ]
%
%  and P_B denotes orthogonal projection onto the range of B
if nargin < 5 || strcmp(clampType,'L2'),
    if nargin > 3, 
        [xHat,lambdaHat,totalIt,relRes] = minimize_l1_lc_qc(w,P_B_handle,r,xCur,eta,lambda);
    else
        [xHat,lambdaHat,totalIt,relRes] = minimize_l1_lc_qc(w,P_B_handle,r,xCur,eta);
    end
else
    [xHat,lambdaHat,totalIt,relRes] = minimize_l1_lc_ic(w,P_B_handle,r,xCur,eta,lambda);
end

deltaA = reshape(xHat(1:mn),m,n);
deltaX = reshape(xHat(mn+1:end),n,p)-X;