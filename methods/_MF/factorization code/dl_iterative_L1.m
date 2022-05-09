function [A,X] = dl_iterative_L1(Y,A0,X0)

% dl_iterative_L1
%
%   Seek a local optimum for the problem
%
%      min ||X||_1  subj  Y = AX,  ||A_i|| = 1 for all i
%
%   Inputs:
%      Y  -  input matrix  m x p
%      n  -  number of basis vectors
%
%   Outputs: 
%      A  -  sparsifying basis
%      X  -  sparse coefficients
%
%   Winter 2009, John Wright. Questions? jowrig@microsoft.com
%
%   Modified 10/23/10 by JW to handle rescaling in a more sensible manner. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE        = 2;
ALPHA          = 0.05;
BETA           = 0.5;
MAX_ITERATIONS = 300; 
MAX_LINE_SEARCH_ITERATIONS = 25;
MAX_STEP_SIZE_TO_TERMINATE = 1e-7;
MAX_ETA_TO_TERMINATE = 1e-4;
RESCALE        = true;
EPS_CUT        = 1e-2;
CLAMP_TYPE     = 'L2';
LAMBDA_INIT    = 10000;
ETA_INIT       = 2;

[m,p] = size(Y);
    A = A0;
    X = X0;

eta = ETA_INIT;
lambda = LAMBDA_INIT;
converged = false;
numIterations = 0;
converged = false; 
displayScale = 1;


while ~converged,   
       
    numIterations = numIterations + 1;
    
    normA = norm(A);
    normX = norm(X); 
    
    if mod(numIterations,5) == 1,
        Y = Y * (1 / normX);
        X = X * (1 / normX);
        displayScale = displayScale * (normX / 1);
    end        
    
    if VERBOSE > 1,
        disp('L1 Minimization over the tangent space');
    end
    
    lambdaValue(numIterations) = lambda;
    
    % L1 minimization over T_p M
    %                                                                      
    %    Solve the minimization problem                                    
    %                                                                      
    %    min_{DA,DX}  || X + DX ||_1   subj   A DX + DA X = 0              
    %                                           A_i' DA_i = 0, i = 1 ...   
    [DA,DX,lambda,numIt,relRes] = tangent_space_L1_alm(A,X, eta, lambda, CLAMP_TYPE);  
    %[DA,DX] = tangent_space_L1_ip_clamped(A,X,eta); %_clamped(A,X,eta);
    %lambda = LAMBDA_INIT;
    %relRes = 0;
    %numIt = 0;
         
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
            disp(['   Line search iteration ' num2str(numLineSearchIter) '  t: ' num2str(t) '  ||X||_1:  ' num2str(oldObj) '  ||X_new||_1: ' num2str(newObj) '  Residual ||Y-A_new*X_new||_F:  ' num2str(norm(Y-A_new*X_new)) ]);
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
        disp(['Iteration ' num2str(numIterations) '  eta: ' num2str(eta) '  ||X||_1: ' num2str(displayScale * sum(sum(abs(X)))) '  Total step: ' num2str(totalStep) '  Rel res: ' num2str(relRes(end)) '  lambda: ' num2str(lambda) '  numIt: ' num2str(numIt) '  pdist: ' num2str(bestProjDist) '  Step ' num2str(sqrt(norm(DA,'fro')^2 + norm(DX,'fro')^2))]);
    end
  
    if totalStep >= eta / 3, 
        eta = 1.075 * eta;
    elseif totalStep < eta / 3,
        eta = eta / 2; 
    end
    
    if numIterations >= MAX_ITERATIONS || (totalStep <= MAX_STEP_SIZE_TO_TERMINATE && eta < MAX_ETA_TO_TERMINATE), 
        converged = true;
    end
end

X = displayScale * X;

