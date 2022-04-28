function [resY , resX] = CoROMP3(D , Y , X , param)
% Modified Orthogonal Matching Pursuit
%   Based on a CoSaMP_SB with Two_Solve and HSS
%
% Inputs :
% D     : dictionary (normalized columns)
% Y     : set of vectors to run on (each column is one signal)
% X     : Initial set of coefficients (one column for one signal)
% param : stopping condition, containing at least one of these rows
%         * 'errorGoal' and 'noiseSig'
%         * 'maxAtoms'
%
% Outputs :
% resY  : The result vectors
% resX  : Sparse coefficient matrix

% Get parameters and allocate result
dim = size(D , 1);
nAtoms = size(D , 2);
nSignals = size(Y , 2);
initX = nnz(X) && size(X,2) == size(Y,2);
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

fprintf('CoROMP3; card,blksz,incrAdd,addK,addX,gain,sigma,epsilon,initX,coeffThres,maxAtoms = \n  %g,  %g,  %g,  %g,  %g,  %g,  %g,  %g, %g, %g, %g  \n', ...
    card,blksz,incrAdd,addK,addX,gain,sigma,epsilon,initX,coeffThres,maxAtoms);

% Allocate vectors to insert coefficients into
% We keep them as triplets (signalInd, atomInd, coeffVal)
% In the end, we will construct a sparse matrix from them, as this is more efficient
allCoeffsList  = zeros(1 , nSignals * min(card , ceil(1*nAtoms)));
allindsList    = zeros(size(allCoeffsList));
allSignalsList = zeros(size(allCoeffsList));
totNcoeffs = 0;

itAvg = 0;
% chg = 0;
b0 = zeros(nAtoms,1);
K0 = 0;
x = zeros(nAtoms,1);
% Run loop on signals
for cSignal = 1 : nSignals
    
    % process one signal
    y = Y(: , cSignal);
    x(:) = 0;
    K = 0;
    indsList = [];
    residual = y;
    lastK = 0;
    lastT = [];

    if initX
        x1 = X(: , cSignal);
        K = round(nnz(x1)/3);
        if K == 0
            K = nnz(x1);
        end
%         if testSim && cSignal > 1 
%             K = max(0, param.Kold - fix(simFactor*(1 - similarity(cSignal-1)))); 
%         end
        if errConstraint
            mx = addX*coeffThres;
            indsList = find(abs(x1(indsList))>=mx);
            K = min(length(indsList),round(card/2));
        else
            if incrAdd
                mx = addX*max(abs(x1));
                indsList = find(abs(x1(indsList))>=mx);
                K = min(length(indsList),round(card/2));
            else
                [~,indsList] = sort(abs(x1),'descend');
                indsList = indsList(1:K);
            end
        end
        K0 = K0 + K;
        if ~isempty(indsList)
            x(indsList) = D(: , indsList) \ y;
            residual = y - D(: , indsList) * x(indsList);
            lastK = length(indsList);
            lastT = sort(indsList);
        end
    end
    
    if errConstraint && norm(residual) < epsilon
        continue
    end
    
    % Repeat as long as residual decreases
    % while residNorm < old_RN
    %   Changed to repeat while residual is changing.
    for it = 1:maxIter
        K = min(K+addK,card);
        
        % Compute projections
        proj = D' * residual;
        
        % Find the index of the largest absolute inner product
%       The following replaces:  T = union(indsList,  maxPrInd(1:addK)');
%       Because the union function took more than 10x longer.
        b0(:) = 0;
        b0(indsList) = 1;
        if addK > 1
            [~ , maxPrInd] = sort(abs(proj),'descend');
            b0(maxPrInd(1:addK)) = 1;
        else
            [~ , maxPrInd] = max(abs(proj));
            b0(maxPrInd) = 1;
        end
        T = find(b0);
        
        % Compute new coefficients with this combined indsList
        if HSS
            b = D(:,T) \ residual;
            x0 = x;
            x0(T) = b + x0(T);
            if errConstraint
                T = find(abs(x0)>coeffThres);
                K = length(T);
            else
                [~,maxInd] = sort(abs(x0),'descend');
            end
        else
            b = D(:,T) \ y;
            if errConstraint
%                 b(abs(b)<coeffThres) = 0;
%                 T = T(b~=0);
                T = T(abs(b)>coeffThres);
                K = length(T);
            else
                [~,maxInd] = sort(abs(b),'descend');
            end
        end
        if errConstraint
            if K >= maxAtoms
                break;
            elseif K==lastK 
                T = sort(T);
                if K==0 || nnz(T==lastT) == K
                    break;
                end
            end
        else
            K = min(K,length(T));
            T = maxInd(1:K);
        end
        
        % Compute new residual
        b = D(:,T) \ y;
        oldR = residual;
        residual = y - D(: , T) * b;
        
%   Try changing the LS comp to compute the change in the residual, i.e.,
%   residual = residual + droppedCoeffs - addedCoeffs
        
        %             residNorm = sum(residual.^2);
        %             iter = iter + 1;
        if ~errConstraint && sum( (residual - oldR).^2 ) < 1e-2
            break;
        end
        x(:) = 0;
        indsList = T;
        x(indsList) = b;
        lastK = K;
        lastT = sort(indsList);
    end
    %         chg = chg  + (card - length(intersect(indsList,oldSup)));
    indsList = find(x);
    coeffsList = x(indsList);
    K = length(indsList);
    
    % Assign results
    allCoeffsList( totNcoeffs + (1 : K)) = coeffsList(1 : K);
    allindsList(   totNcoeffs + (1 : K)) = indsList(1 : K);
    allSignalsList(totNcoeffs + (1 : K)) = cSignal;
    totNcoeffs = totNcoeffs + K;
    itAvg = itAvg + it;
end

% Construct the sparse matrix
resX = sparse(...
    allindsList(1:totNcoeffs) , allSignalsList(1:totNcoeffs) , allCoeffsList(1:totNcoeffs) , ...
    nAtoms , nSignals);

% Create the output signals
resY = D * resX;

% Finished
fprintf('CoROMP3; iterations %g, ', ...
    itAvg/nSignals);
fprintf(' avg K= %g, Total K= %g avg old coeffs used= %g \n',totNcoeffs/nSignals,totNcoeffs,K0/nSignals);
return;

end


