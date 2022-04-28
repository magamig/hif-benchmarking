function [resX , resA] = Batch_OMP(D , X , param)
% Run Matching Pursuit
%
% Inputs : 
% D     : dictionary (normalized columns)
% X     : set of vectors to run on (each column is one signal)
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

% determine stopping criteria
testErrorGoal = false; errorGoal = inf; 
if isfield(param , 'errorGoal'), 
	testErrorGoal = true; 
	
	% Compute the actual error goal, and account for noise and signal length
	errorGoal = param.errorGoal * dim * (param.noiseSig.^2); 
end;
testMaxAtoms = false; maxAtoms = nAtoms;
if isfield(param , 'maxAtoms'), testMaxAtoms = true; maxAtoms = param.maxAtoms; end;
if (~testErrorGoal) && (~testMaxAtoms), error('At least one stopping criterion is needed!'); end;

% Allocate vectors to insert coefficients into
% We keep them as triplets (signalInd, atomInd, coeffVal)
% In the end, we will construct a sparse matrix from them, as this is more efficient
allCoeffsList  = zeros(1 , nSignals * min(maxAtoms , ceil(0.2*nAtoms)));
allIndsList    = zeros(size(allCoeffsList));
allSignalsList = zeros(size(allCoeffsList));
totNcoeffs = 0;

% Compute DtD and DtX
DtD = D' * D; 
DtX = D' * X; % This might not work for a large number of signals. 
              % It is usedful to break X into groups of signals in that case
			  % Alternatively, this can be computed for each signal, however, this is slower
sigSumSqrs = sum(X.^2 , 1);

% Run loop on signals
if (nSignals > 1000), fprintf('OMP (thousands) : '); end;
for cSignal = 1 : nSignals
	
	if (nSignals > 1000) && (mod(cSignal , 1000) == 0), fprintf('%d ' , cSignal / 1000); end;
	
	% process one signal
	
	% get current signal - get its inner products with the dictionary directly (D^T x)
	initInnerProductsList = DtX(: , cSignal);
	currInnerProductsList = initInnerProductsList;
	
	% init the residual size counter
	residSumSqrs = sigSumSqrs(cSignal);
	
	% This is used for updating the resiudal norm at each stage
	prev_delta = 0;
	
	% Make sure the initial signal in itself has enough energy, 
	% otherwise the zero signal is returned
	if ((testErrorGoal) && (residSumSqrs < errorGoal))
		continue; % simply move on to the next signal
	end
	
	% Initialize indices vectors
	indsList = [];
	card = 0;
	
	% Repeat as long as stopping condition isn't filled
	while 1
		
		% Update cardinality
		card = card + 1;
		
		% Find the index of the largest absolute inner product
		[~, maxProjInd] = max(abs(currInnerProductsList));
		
		% If this is the first atom, keep its projection
		if (card == 1)
			coeffsList = currInnerProductsList(maxProjInd);
			indsList = maxProjInd;
		else
			% If not the first atom, do least-squares (LS) over all atoms in the representation
			indsList = [indsList maxProjInd];
			LS_LHS = DtD(indsList , indsList); % The left hand side of the LS equation to solve
			LS_RHS = initInnerProductsList(indsList);
			coeffsList = LS_LHS \ LS_RHS;
		end
		
		% Update the inner products list
		beta = DtD(: , indsList) * coeffsList;
		currInnerProductsList = initInnerProductsList - beta;
		
		% Check if we can stop running
		if testErrorGoal % Error Treshold
			
			% We only need to update the residual computation if the stopping criterion is the error 
			curr_delta = sum(coeffsList .* beta(indsList));
			residSumSqrs = residSumSqrs + prev_delta - curr_delta;
			prev_delta = curr_delta;
			
			if residSumSqrs < errorGoal, break; end;
		end
		
		if testMaxAtoms % Cardinality Threshold
			if (card >= maxAtoms), break; end;
		end
		
	end
		
	% Assign results
	allCoeffsList( totNcoeffs + (1 : card)) = coeffsList;
	allIndsList(   totNcoeffs + (1 : card)) = indsList(1 : card);
	allSignalsList(totNcoeffs + (1 : card)) = cSignal;
	totNcoeffs = totNcoeffs + card;
end

if (nSignals > 1000), fprintf('\n'); end;
	
% Construct the sparse matrix
resA = sparse(...
	allIndsList(1:totNcoeffs) , allSignalsList(1:totNcoeffs) , allCoeffsList(1:totNcoeffs) , ...
	nAtoms , nSignals);

% Create the output signals
resX = D * resA;

% Finished
return;