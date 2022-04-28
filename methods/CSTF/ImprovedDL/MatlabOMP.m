function [resX , resA] = MatlabOMP(D , X , param)
% Run Orthogonal Matching Pursuit
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
precision = 1e-8;
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
allCoeffsList  = zeros(1 , nSignals * min(maxAtoms , ceil(1*nAtoms)));
allIndsList    = zeros(size(allCoeffsList));
allSignalsList = zeros(size(allCoeffsList));
totNcoeffs = 0;

% Run loop on signals
for cSignal = 1 : nSignals
	% process one signal
	
	% Get current signal
	x = X(: , cSignal);
	
	% Initialize coefficient and indices vectors
	residual = x;
	coeffsList = zeros(dim , 1);
	indsList = zeros(1 , dim);
	card = 0;
	
	% Make sure the initial signal in itself has enough energy, 
	% otherwise the zero signal is returned
	if ((testErrorGoal) && (sum(residual.^2) < errorGoal))
		continue; % simply move on to the next signal
	end
	
	% Repeat as long as stopping condition isn't filled
	while 1
		
		% Update cardinality
		card = card + 1;
		
		% Compute projections
		proj = D' * residual;
		
		% Find the index of the largest absolute inner product
		[~, maxProjInd] = max(abs(proj));
				
		% Add this new atom to the list
		indsList(card) = maxProjInd;
		
		% If this is the first atom, keep its projection
		if (card == 1)
			coeffsList(card) = proj(maxProjInd);
		else
			% If not the first atom, do least-squares (LS) over all atoms in the representation
			coeffsList(1 : card) = D(: , indsList(1 : card)) \ x;
		end
		
		% Compute new residual
		residual = x - D(: , indsList(1 : card)) * coeffsList(1 : card);
		
		% Check if we can stop running
        residNorm = sum(residual.^2);
        if (residNorm < precision), break; end; 
		if testErrorGoal % Error Treshold
			if residNorm < errorGoal, break; end;
		end
		
		if testMaxAtoms % Cardinality Threshold
			if (card >= maxAtoms), break; end;
		end
		
    end
		
	% Assign results
	allCoeffsList( totNcoeffs + (1 : card)) = coeffsList(1 : card);
	allIndsList(   totNcoeffs + (1 : card)) = indsList(1 : card);
	allSignalsList(totNcoeffs + (1 : card)) = cSignal;
	totNcoeffs = totNcoeffs + card;
end
	
% Construct the sparse matrix
resA = sparse(...
	allIndsList(1:totNcoeffs) , allSignalsList(1:totNcoeffs) , allCoeffsList(1:totNcoeffs) , ...
	nAtoms , nSignals);

% Create the output signals
resX = D * resA;

% Finished
return;
