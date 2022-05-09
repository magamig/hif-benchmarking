function [x_hat, numIterations] = minimize_l1_l2_qc(w,A,x0,b,lambda,y,eta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Solves the optimization problem
% 
%    min ||x||_{w,1} + lambda ||Ax - b||_2^2 
%    subject to ||x-y||_2 <= eta
%
%  Uses a first order method similar to iterative thresholding. 
%
%  THIS VERSION ASSUMES THE OPERATOR A IS SQUARE AND IDEMPOTENT, I.E.,
%    A IS THE ORTHOPROJECTOR ONTO SOME SUBSPACE S <= Re^m  
%
%  Inputs:
%    w  - m x 1 nonnegative vector (can include zeros), defining the
%                weighted L1 norm
%    A  - m x m matrix, or function handle
%    x0 - m x 1 vector, initial guess, should fall within distance eta of y
%    b  - m x 1 vector, observation
%    y  - m x 1 vector
%    eta - scalar, maximum distance from y
%
%  Outputs:
%    x_hat - n x 1 vector, optimal solution
%
%  Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE = 0;
MAX_ITERATIONS = 1500;
MIN_STEP = 5e-5 * eta;
delta    = .95;

implicitMode = isa(A,'function_handle');

m = length(b);
n = length(x0);

converged = false;
numIterations = 0;

x = x0;

clear('x0');

while ~converged,
    
    numIterations = numIterations + 1;
    
    % gradient step
    if implicitMode,
        g = 2 * A( x - b );
    else
        g = 2 * A * ( x - b );
    end
    
    % shrinkage step
    %   minimize     ||x||_1 + ( lambda / 2 delta ) ||x - (xPrev - delta g)||^2
    %   subject to   ||x - y|| <= eta
    xNew = constrained_shrinkage_2(x - delta * g,y,w,lambda / (2*delta),eta);
    
    stepSize = norm(xNew-x);
    x = xNew; 
    
    % output
    if VERBOSE,
        disp(['   Iteration ' num2str(numIterations) '  ||x||_1 ' num2str(w' * abs(x)) '  step ' num2str(stepSize) '  ||x-y|| ' num2str(norm(x-y))]);
    end
    
    % check convergence
    if numIterations >= MAX_ITERATIONS || stepSize < MIN_STEP
        converged = true;
    end
end

x_hat = x;