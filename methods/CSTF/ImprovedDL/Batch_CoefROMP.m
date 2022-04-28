function [resX , resA] = Batch_CoefROMP(D , X , Gamma , G , param)
% Run Matching Pursuit
%
% Inputs : 
% D     : dictionary (normalized columns)
% X     : set of vectors to run on (each column is one signal)
% Gamma : set of initial coefficients (each column corresponds to a signal)
% param : stopping condition, containing at least one of these rows 
%         * 'errorGoal' and 'noiseSig'
%         * 'maxAtoms'
%
% Outputs :
% resX  : The result vectors
% resA  : Sparse coefficient matrix

% Get parameters and allocate result
dim = size(D , 1);
nAtoms = size(D , 2);
nSignals = size(X , 2);
initGamma = nnz(Gamma) && size(Gamma,2) == size(X,2);
blksz = round(sqrt(dim));

%  Initializations
function out = setParam( field, default )
    if ~isfield( param, field )
        param.(field)    = default;
    end
    out = param.(field);
end

HSS = setParam( 'HSS', 1 );
errConstraint = setParam( 'errConstraint', 0 );
incrAdd  = setParam( 'incrAdd', 0 );
addK  = setParam( 'addK', 1 );
addX  = setParam( 'addX', 2 );
card  = setParam( 'card', round(dim/10) );
maxIter  = setParam( 'maxIter', 40 );
maxAtoms  = setParam( 'maxAtoms', round(dim/4) );
coeffThres  = setParam( 'coeffThres', mean(X(X~=0)) );
gain  = setParam( 'gain', 1.125 );
sigma  = setParam( 'sigma', 10 );
epsilon  = setParam( 'epsilon', gain*blksz*sigma );

fprintf('CoefROMP; card,blksz,incrAdd,addK,addX = %g,  %g,  %g,  %g,  %g \n', ...
    card,blksz,incrAdd,addK,addX);
fprintf('  gain,sigma,epsilon,initGamma,coeffThres,maxAtoms,errConstraint = %g,  %g,  %g,  %g,  %g,  %g,  %g  \n', ...
    gain,sigma,epsilon,initGamma,coeffThres,maxAtoms,errConstraint);

% determine stopping criteria
% testErrorGoal = false; errorGoal = inf; 
% if isfield(param , 'errorGoal'), 
% 	testErrorGoal = true; 
% 	
% 	% Compute the actual error goal, and account for noise and signal length
% 	errorGoal = param.errorGoal * dim * (param.noiseSig.^2); 
% end;
% testMaxAtoms = false; maxAtoms = nAtoms;
% if isfield(param , 'maxAtoms'), testMaxAtoms = true; maxAtoms = param.maxAtoms; end;
% if (~testErrorGoal) && (~testMaxAtoms), error('At least one stopping criterion is needed!'); end;

% Allocate vectors to insert coefficients into
% We keep them as triplets (signalInd, atomInd, coeffVal)
% In the end, we will construct a sparse matrix from them, as this is more efficient
allCoeffsList  = zeros(1 , nSignals * min(card , ceil(0.2*nAtoms)));
allIndsList    = zeros(size(allCoeffsList));
allSignalsList = zeros(size(allCoeffsList));
totNcoeffs = 0;

% Compute DtD and DtX
if G
    DtD = G;      %  D' * D; 
else
    DtD = D' * D; 
end
DtX = D' * X; % This might not work for a large number of signals. 
              % It is usedful to break X into groups of signals in that case
			  % Alternatively, this can be computed for each signal, however, this is slower
Tol = 1e-4*sum(abs(DtX));

% if XtX 
%     sigSumSqrs = XtX;
% else
%     sigSumSqrs = sum(X.^2 , 1);
% end

beta = zeros(nAtoms,1);
b0 = zeros(nAtoms,1);
% T = zeros(card+addK,1);
g = zeros(nAtoms,1);
K0 = 0;
itAvg = 0;
% Run loop on signals
% if (nSignals > 1000), fprintf('Batch_CoROMP (thousands) : '); end;
for cSignal = 1 : nSignals
	
% 	if (nSignals > 1000) && (mod(cSignal , 1000) == 0), fprintf('%d ' , cSignal / 1000); end;
	
	% process one signal
	
	% init the residual size counter
% 	residSumSqrs = sigSumSqrs(cSignal);
		
    % get current signal - get its inner products with the dictionary directly (D^T x)
    initInnerProductsList = DtX(: , cSignal);   %   alpha0 in paper
    
    % Residual computation used for stopping condition (delta is update
    % of residual norm)
    curr_delta = 0;
    
    g(:) = 0;
    K = 0;
    beta(:) = 0;
    indsList = [];
    lastK = 0;
    lastT = [];
    if initGamma
%         indsList = find(Gamma(:,cSignal));
        g1 = Gamma(:,cSignal);
        K = round(nnz(g1)/3);
        if K == 0
            K = nnz(g1);
        end
        
        %   Use only some of the previous Gammas
        if errConstraint
            mx = addX*coeffThres;
            indsList = find(abs(g1(indsList))>=mx);
            K = min(length(indsList),round(card/2));
        else
            if incrAdd
                mx = addX*max(abs(g1));
                indsList(abs(g1(indsList))<mx) = 0;
                indsList = indsList(indsList>0);
                K = length(indsList);
            else
                [~,indsList] = sort(abs(g1),'descend');
                indsList = indsList(1:K);
            end
        end
        K0 = K0 + K;
            
        if ~isempty(indsList)
        %   Recompute Gamma from dictionary and signal; makes results
        %   worse
            g(indsList) = DtD(indsList,indsList) \ DtX(indsList , cSignal);
        
        % Update the inner products list (Product of DtD * Gamma)
            beta = DtD(: , indsList) * g(indsList);
            curr_delta = sum(g(indsList) .* beta(indsList));
             if errConstraint
                lastK = length(indsList);
                lastT = sort(indsList);
             end
        end
    end

%     if errConstraint && norm(curr_delta) > epsilon
%         continue      %   If error is small, skip loop and go to the next signal
%     end
        
    % Repeat as long as residual decreases
    for it = 1:maxIter
        % Update the inner products list (alpha in paper)
        currInnerProductsList = initInnerProductsList - beta;

%         if sum(abs(currInnerProductsList)) < Tol(cSignal)
%             break;
%         end

        % Find the indices of the largest absolute inner product
        if addK > 1
            [~, maxProjInd] = sort(abs(currInnerProductsList),'descend');
        else
            [~, maxProjInd] = max(abs(currInnerProductsList));
        end

        %   Combine max projection indicies with current support
%       The following replaces:  T = union(support,  maxPrInd(1:addK)');
%       Because the union function took more than 10x longer.
        b0(:) = 0;
        b0(indsList) = 1;
        b0(maxProjInd(1:addK)) = 1;
        T = find(b0);
        K = min(length(T),card+addK);
        T = T(1:K);
        
        % Compute new coefficients with this combined support
        LS_LHS = DtD(T , T); % The left hand side of the LS equation to solve
        if HSS        %   Revised to include HSS option
            LS_RHS = currInnerProductsList(T);
            b = LS_LHS \ LS_RHS;
            b0(T) = g(T) + b;
        else
            LS_RHS = initInnerProductsList(T);
            b = LS_LHS \ LS_RHS;
            b0(T) = b;
        end
        if errConstraint
            T = find(abs(b0)>coeffThres);
            K = length(T);
            if K >= maxAtoms
                break;
            elseif K==lastK 
                T = sort(T);
                if K==0 || nnz(T==lastT) == K
                    break;
                end
            end
        else
            K = min(K,card);
            [~,maxInd] = sort(abs(b0),'descend');
            T = maxInd(1:K);
%             [~, T] = nLargest(K,b0);
        end

        %   Recompute Gamma with current set of coefficients
        b = DtD(T , T) \ initInnerProductsList(T);

        % Update the beta list (Product of DtD * Gamma)
        beta = DtD(: , T) * b;

        prev_delta = curr_delta;
        curr_delta = sum(b .* beta(T));
        
        %   Quit if no improvement in error
        if ~errConstraint && (curr_delta-prev_delta) < 0.01
            break;
        end
        g(:) = 0;
        indsList = T;
        g(indsList) = b;
        if errConstraint
            lastK = K;
            lastT = sort(indsList);
        end
    end
    indsList = find(g);
    coeffsList = g(indsList);
    K = length(indsList);

	% Assign results
	allCoeffsList( totNcoeffs + (1 : K)) = coeffsList(1 : K);
	allIndsList(   totNcoeffs + (1 : K)) = indsList(1 : K);
	allSignalsList(totNcoeffs + (1 : K)) = cSignal;
	totNcoeffs = totNcoeffs + K;
    itAvg = itAvg + it;
end

% if (nSignals > 1000), fprintf('\n'); end;
	
% Construct the sparse matrix
resA = sparse(...
	allIndsList(1:totNcoeffs) , allSignalsList(1:totNcoeffs) , allCoeffsList(1:totNcoeffs) , ...
	nAtoms , nSignals);

% Create the output signals
resX = D * resA;

% Finished
fprintf('Batch-CoefROMP iterations %g, card= %g, incrAdd,initX =  %g, %g  \n',itAvg/nSignals,card,incrAdd,initGamma);
fprintf(' avg K= %g, Total K= %g avg old coeffs used= %g, HSS=%g \n', ...
    totNcoeffs/nSignals,totNcoeffs,K0/nSignals,HSS);
return;
end

function [values, indicies] = nLargest(number,coeffs)

values = coeffs(1:number);
indicies = 1:number;
[minV,indx] = min(values);

for i=number+1:length(coeffs)
    if coeffs(i) > minV
        values(indx) = coeffs(i);
        [minV,indx] = min(values);
    end
end

end