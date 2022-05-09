function [A,X] = dl_iterative_L1_psg(Y,A0,X0)
    
% dl_iterative_L1_psg.m
%
%   Iteratively minimize the L1 norm via a projected subgradient method
%
%
%   December 2010, John Wright
%   Questions? jowrig@microsoft.com

VERBOSE        = 1;
ALPHA          = 0.05;
BETA           = 0.5;
MAX_ITERATIONS = 10000; 
MAX_LINE_SEARCH_ITERATIONS = 25;
MAX_STEP_SIZE_TO_TERMINATE = 1e-7;
MAX_ETA_TO_TERMINATE = 1e-4;

[m,p] = size(Y);
[n,p] = size(X0);

A = A0;
X = X0;

converged = false;
numIterations = 0;

while ~converged,   
       
    numIterations = numIterations + 1;
    
    B_info = P_B_precomputations_square_new(A,X);    
    h = [ zeros(m*n,1); vec(sign(X) .* ( (abs(X) > 1e-2) + .5 * (abs(X) > 1e-3) + .25 * (abs(X) > 1e-4) ) / 1.75 )   ];    
    h_proj = P_null_B_square_new(h,B_info);
    
    DA = - reshape(h_proj(1:m*n),m,n);
    DX = - reshape(h_proj(m*n+1:end),n,p);
        
    if VERBOSE > 1, 
        disp('Line search ...');
    end
    
    % line search
    line_search_converged = false;
    numLineSearchIter = 0;
    
    t = 1;
    oldObj = sum(sum(abs(X)));
    
    best_t  = 2;
    bestObj = oldObj;
    bestA   = A;
    bestX   = X;
    
    bestProjDist = inf;
             
    while ~line_search_converged,        
        
        numLineSearchIter = numLineSearchIter + 1;
        
        A_new = A + t * DA;
        X_new = X + t * DX;
        
        A_new_preproj = A_new;
        X_new_preproj = X_new;
        
        [A_new,X_new] = naive_projection(Y,A_new,X_new);
        
        projDist = sqrt(norm(A_new - A_new_preproj,'fro')^2 + norm(X_new - X_new_preproj,'fro')^2);
                
        newObj = sum(sum(abs(X_new)));
        
        if newObj < bestObj,
            bestObj = newObj;
            best_t = t;
            bestA = A_new;
            bestX = X_new;
            bestProjDist = projDist;
        end        
        
        if VERBOSE > 1,
            disp(['   Line search iteration ' num2str(numLineSearchIter) '  t: ' num2str(t) '  ||X||_1:  ' num2str(oldObj) '  ||X_new||_1: ' num2str(newObj) '  Residual ||Y-A_new*X_new||_F:  ' num2str(norm(Y-A_new*X_new))]);
        end
        
        if ( newObj <= (1-ALPHA*t)*oldObj || numLineSearchIter >= MAX_LINE_SEARCH_ITERATIONS )   
            
            A_new = bestA;
            X_new = bestX;
            
            A_step = norm(A_new-A,'fro');
            X_step = norm(X_new-X,'fro');
            totalStep = sqrt(A_step*A_step + X_step*X_step);
                        
            A = A_new;
            X = X_new; 
            line_search_converged = true;
        else
            t = BETA * t;
        end
    end
            
    if VERBOSE,
        disp(['Iteration ' num2str(numIterations) '  ||X||_1: ' num2str(sum(sum(abs(X)))) '  Total step: ' num2str(totalStep) '  pdist: ' num2str(bestProjDist) '  best_t: ' num2str(best_t)]);
    end
    
    if numIterations >= MAX_ITERATIONS || (totalStep <= MAX_STEP_SIZE_TO_TERMINATE) % && eta < MAX_ETA_TO_TERMINATE), 
        converged = true;
    end
end

%X = displayScale * X;
