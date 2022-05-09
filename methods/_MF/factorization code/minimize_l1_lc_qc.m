function [x_hat, lambdaHat, totalIterations, relativeResidual] = minimize_l1_lc_qc(w,A,b,y,eta,lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Solves the optimization problem
%
%     minimize ||x||_{w,1}  subject  Ax = b, ||x-y|| <= eta
%
%   Currently assumes that Ay = b, so the feasible set is nonempty. Uses
%   Bregman iteration / ALM together with an iterative thresholding
%   algorithm for solving the inner iterations. 
%
%  THIS VERSION ASSUMES THE OPERATOR A IS SQUARE AND IDEMPOTENT, I.E.,
%    A IS THE ORTHOPROJECTOR ONTO SOME SUBSPACE S <= Re^m  
%
%   Inputs:
%      w      - m x 1 nonnegative weight vector, defines the norm
%      A      - m x m matrix, or function handle
%      x0     - m x 1 initial guess, should be feasible
%      b      - m x 1 observation vector
%      y      - m x 1 clamp
%      eta    - scalar distance from y
%      lambda - optional scalar parameter, the relaxation used in forming
%               the augmented Lagrangian
%
%   Outputs:
%      x_hat     - solution
%      lambdaHat - a guess at a new lambda that would have resulted in
%                   faster solution. 
%
%   Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE                         = 0;
MAX_ITERATIONS                  = 80000; %5000; % was 2500;
DEFAULT_LAMBDA                  = 50; 
MIN_INNER_IT_TO_DECREASE_LAMBDA = 400;
TERMINATE_IF_NONMONOTONE        = 1; 
RELATIVE_RESIDUAL_THRESHOLD     = 1e-4;

if nargin < 6,
    lambda = DEFAULT_LAMBDA;
end

implicitMode = isa(A,'function_handle');

m = length(w);
r = zeros(m,1);
x = zeros(m,1);

converged = false;
prevResidual = inf; 
minInnerItReached = false;
numIterations = 0;
totalIterations = 0;

delta = b - A(x);
normb = norm(b);

while ~converged,
    numIterations = numIterations + 1;
    
    if implicitMode,
        r = r + delta;
    else
        r = r + delta;
    end

    [xNew, it] = minimize_l1_l2_qc(w,A,x,r,lambda,y,eta);
    
    if ( it >= MIN_INNER_IT_TO_DECREASE_LAMBDA ),
        minInnerItReached = true;
    end
    
    totalIterations = totalIterations + it;
    
    stepSize = norm(xNew-x);
    x = xNew;
    
    if implicitMode,
        delta = b - A(x);
    else
        delta = b - A*x;
    end    
    
    residual = norm(delta);
    relativeResidual = residual / eta; 
    
    if VERBOSE,
        disp(['Bregman iteration ' num2str(numIterations) '  Total it: ' num2str(totalIterations) '  Objective ' num2str(w' * abs(x)) '  Residual ' num2str(residual) '  Relative residual ' num2str(relativeResidual)]);
    end
    
    if totalIterations >= MAX_ITERATIONS || (relativeResidual < RELATIVE_RESIDUAL_THRESHOLD) || (residual >= prevResidual && TERMINATE_IF_NONMONOTONE)      
        converged = true;
    end
    
    prevResidual = residual;
end

x_hat = x;

if minInnerItReached,
    lambdaHat = lambda / 2;
elseif ( numIterations >= 75 && totalIterations >= 1500 ) || (relativeResidual > RELATIVE_RESIDUAL_THRESHOLD),
    lambdaHat = lambda * 2;
else
    lambdaHat = lambda;
end