function [resDict , allDicts] = Train_Dictionary(trainPatches, param)
% Train a dictionary using either K-SVD or MOD
%
% Inputs : 
% trainPatches : The set of patches to train on
% param        : various parameters, continaing the fields (fields with defaul indicated are optional)
%                * 'method'          : 'KSVD' or 'MOD'
%                * (at least) One of ('errorGoal' & 'noiseSig') or 'maxAtoms' : 
%                                      stopping condition for the OMP
%                * 'nAtoms'          : number of atoms in the dictionary
%                * 'nIterations'     : number of dictionary update iterations
%                * 'initType'        : 'RandomPatches' to take a random set of patches, 
%                                      'DCT' for an overcomplete DCT dictionary (assumes patch is square)
%                                      'Input' an initial dictionary is given
%                * 'initDict'        : Only if param.initType == 'Input' - the initial dictionary to use
%                * 'maxIPforAtoms'   : The maximal allowed inner profuct between two atoms 
%                                      (if two atoms are created with larger IP, one is replaced).
%                                      Default: 0.99
%                * 'minTimesUsed'    : The minimal number of times an atom should have a meaningful coefficient.
%                                      If less or equal to that, it is replaced
%                                      Default: 3 
%                * 'meaningfulCoeff' : The smallest value that a coefficient is considered to be different than 0.
%                                      Default: 10^-7
%                * 'showDictionary'  : After how many iterations show the dictionary (0 to no-show)
%                                      Default: 0
%                * 'patchSize'       : Needed for showing the dictionary
%                * 'trueDictionary'  : Optional. Needed to estimate how close the recovered dictionary is to true one
%                * 'atomRecoveredThresh' : Needed if 'trueDictionary' is provided. 
%                                      Atom from trueDictionary is considered to be recovered 
%                                      if there's an atom in the current dictionary, 
%                                      the absolute inner product of these atoms is above this
%                * 'truePatches'     : Optional. The ground truth patches (for computing representation error at each
%                                      iteration)
%                * 'imageData'       : Optional struct, if the dictionary is used for image denoising. 
%                                      If avaliable, uses the dictionary at each stage (except the last) to 
%                                      create an image.
%                                      If the ground truth image is also avaliable, computes the PSNR. 
%                                      It should have the fields:
%                                      'patchSize' (the size of the patch [h w])
%                                      'imageSize' (the size of the original image [h w])
%                                      'withOverlap' (1 or 0, wether patches overlap. If doens't exist, assumes overlap)
%                                      'showImage' wether to show the image in each iteration
%                                      'groundTruthIm' the real image
%         
%
% Outputs :
% ResDict       : The final dictionary
% AllDicts      : A cell array with the dictionaries for each iteration
%                 (first entry is initial, second entry is after first iteration, etc.)

%% Reset random number generator
s = RandStream.getDefaultStream;
reset(s);

%% Get parameters
dim = size(trainPatches , 1);
nAtoms = param.nAtoms;
nPatches = size(trainPatches , 2);

% Default values for some parameters
if ~isfield(param , 'maxIPforAtoms'), param.maxIPforAtoms = 0.99; end;
if ~isfield(param , 'minTimesUsed'), param.minTimesUsed = 3; end;
if ~isfield(param , 'meaningfulCoeff'), param.meaningfulCoeff = 10^-7; end;
if ~isfield(param , 'showDictionary'), param.showDictionary = 1; end;
if param.showDictionary && ~isfield(param , 'patchSize'), 
	error('Must specify patch size for showing dictionary'); 
end;


%% Initialize dictionary
switch param.initType
	case 'RandomPatches' 
		% Select a random sub-set of the patches		
		p = randperm(nPatches);
		currDict = trainPatches(: , p(1 : nAtoms));
		
	case 'DCT'
		% Compute the over-complete DCT dictionary, and remove unnecessary columns and rows
		DCTdict = Build_DCT_Overcomplete_Dictionary(nAtoms , ceil(sqrt(dim)) .* [1 1]);
		currDict = DCTdict(1 : dim , 1 : nAtoms);
		
	case 'Input'
		% Use a given dictionary
		currDict = param.initDict;
		
	otherwise
		error('Unknown dictionary initialization method : %s' , param.initType);
end

% Normalize columns of the dictionary
currDict = currDict ./ repmat(sqrt(sum(currDict.^2, 1)) , size(currDict , 1) , 1);

% store the initial dictionary
allDicts = cell(1 , param.nIterations+1);
allDicts{1} = currDict;

% If ground truth is provided, compute the noisy's quality
if isfield(param , 'truePatches')
	signalErrors = param.truePatches - trainPatches;
	meanSignalsError = mean(sum(signalErrors.^2 , 1));
	fprintf('mean error of noisy signals %02.2f\n' , meanSignalsError);
end

%% Run loops as required
for itr = 1 : param.nIterations
	
	% Do sparse coding - we need the reconstructed patches for the residual
	[currConstructedPatches , currCoeffs] = Batch_OMP(currDict , trainPatches , param);
	
	% Compute average cardinality or representation error, and print information
	if isfield(param , 'errorGoal')
		meanCard = full(sum(abs(currCoeffs(:)) > param.meaningfulCoeff) / nPatches);
		fprintf('Average Card %02.2f\n' , meanCard);
	end
	
	if isfield(param , 'maxAtoms')
		resids = trainPatches - currConstructedPatches;
		meanError = mean(sum(resids.^2 , 1));
		fprintf('mean representation error (compared to training signals) %02.2f\n' , meanError);
	end
	
	% If the true signals are provided, compute the L2 error of their representation
	if isfield(param , 'truePatches')
		signalErrors = param.truePatches - currConstructedPatches;
		meanSignalsError = mean(sum(signalErrors.^2 , 1));
		fprintf('mean recovery error (compared to true signals) %02.2f\n' , meanSignalsError);
	end
	
	
	% If image data is provided, reconstruct image and compute PSNR
	if isfield(param , 'imageData') 
		
		% Create image
		currIm = Average_Overlapping_Patches(currConstructedPatches, ...
			size(param.imageData.imageSize) , param.imageData.patchSize);
		
		% If ground truth is provided, compute the estimate's quality
		if isfield(param.imageData , 'groundTruthIm')
			[currPSNR , currL2] = Compute_Error_Stats(param.imageData.groundTruthIm , currIm);
			fprintf('Current IMAGE PSNR %02.4f, L2 %03.4f\n' , currPSNR , currL2);
		end
		
		% Display image if required
		if param.imageData.showImage
			figure; imshow(currIm); 
			if isfield(param.imageData , 'groundTruthIm')
				title(sprintf('Iteration %d , PSNR = %02.4' , itr - 1, currPSNR));
			else
				title(sprintf('Iteration %d ' , itr - 1));
			end
		end
		
		
		
	end
	
	% Only here the new training iteration starts - 
	% up until here we used the previous dictionary
	fprintf('***********************************\n');
	fprintf('         Iteration %d              \n' , itr);
	fprintf('***********************************\n');
	
	
	% Update dictionary using requested method
	switch param.method
		case 'MOD'
					
			% Compute the new dictionary - at once
			% The addition of (eye(nAtoms) * 10^-7) is for regularization
			% newDict = (trainPatches * currCoeffs') * inv(currCoeffs * currCoeffs' + (eye(nAtoms) * 10^-7));
			
			% Matlab claims the previous equation is inefficient. This should be equivalent but faster
			newDict = (trainPatches * currCoeffs') / (currCoeffs * currCoeffs' + (eye(nAtoms) * 10^-7));
			
			
			% Remove zero vectors by random vectors
			zeroAtomsInds = find(sum(abs(newDict) , 1) < 10^-10);
			newDict(: , zeroAtomsInds) = randn(dim , length(zeroAtomsInds));
			
			% Normalize atoms to norm 1
			newDict = newDict ./ repmat(sqrt(sum(newDict.^2, 1)) , size(newDict , 1) , 1);

		case 'KSVD'
			
			% Compute one new atom at a time 
			% This is incremental (i.e., use previou updated atoms from current iteration too)
			newDict = currDict;
					
			% Run a loop on atoms
			fprintf('Updating Atoms : ' );
			for atomInd = 1 : nAtoms
				
				if mod(atomInd , 10) == 0, fprintf('%d ' , atomInd); end;
				
				% Find all patches using it
				currPatchsInds = find(currCoeffs(atomInd , :));
				
				% Get all coefficients for patches that use the atom
				currPatchsCoeffs = currCoeffs(: , currPatchsInds);
				
				% Set to zero the coefficient of the current atom for each patch
				currPatchsCoeffs(atomInd , :) = 0;
				
				% Compute the residuals for all signals
				resids = trainPatches(: , currPatchsInds) - newDict * currPatchsCoeffs;
				
				% Use svd to determine the new atom, and the new coefficients
				[newAtom , singularVal, betaVec] = svds(resids , 1);
				
				newDict(: , atomInd) = newAtom; % Insert new atom
				currCoeffs(atomInd , currPatchsInds) = singularVal * betaVec'; % Use coefficients for this atom
			end
			fprintf('\n');
			
		otherwise
			error('Unknown dictionary update method %S' , method);
	end
	
	% compute the residual of each signal
	resids = trainPatches - newDict * currCoeffs;
	
	% To improve the dictionary, we now do 2 things:
	% 1. Remove atoms that are rarely used, and replace them by a random entry
	% 2. Remove atoms that have very similar atoms in the dictionary (i.e., large inner product)
	
	% Compute the representation error of each signal
	sigRepErr = sum(resids .^ 2 , 1);
	dictIP = newDict' * newDict; 
	dictIP = dictIP .* (1 - eye(size(dictIP))); % Zero the diagonal - the IP of the atom with itself, so it won't bother us
	
	% Run on each atom, and make the checks
	nReplacesAtoms = 0;
	for cAtom = 1 : nAtoms
		
		maxIPwithOtherAtom = max(dictIP(cAtom , :));
		numTimesUsed = sum(abs(currCoeffs(cAtom , :)) > param.meaningfulCoeff);
		
		if ((maxIPwithOtherAtom > param.maxIPforAtoms) || (numTimesUsed <= param.minTimesUsed))
			
			nReplacesAtoms = nReplacesAtoms + 1;
			
			% Replace the atom with the signal that is worst represented
			[~ , worstSigInd] = max(sigRepErr); worstSigInd = worstSigInd(1);
			newAtom = trainPatches(: , worstSigInd);
			newAtom = newAtom / norm(newAtom);
			newDict(: , cAtom) = newAtom;
			
			% Update the inner products matrix and the representation errors matrix
			sigRepErr(worstSigInd) = 0; % Since it now has an atom for it.
			newIPsForAtom = newDict' * newAtom;
			newIPsForAtom(cAtom) = 0; % The inner product with itself, which we want to ignore
			dictIP(cAtom , :) = newIPsForAtom';
			dictIP(: , cAtom) = newIPsForAtom;
		end
	end
	fprintf('%d atoms replaced for stability\n' , nReplacesAtoms);
	
	
	
	% If we are provided with a ground-truth dictionary, we do two things:
	% Try to order the atoms in the estimated dictionary in the same order as the ground-truth one
	% See how many of the true atoms have been found
	if isfield(param , 'trueDictionary')
		
		% We try to re-order the atoms in the following manner:
		% First, we compute the inner-products between ever true atom and every estimate atom
		% We then go and find the largest IP in this matrix, and determine
		% what true atom and estimated atom achieved it.
		% We then re-order the estimated atom to be in the same place as the original atom
		% and zero both these column and row so both atoms aren't selected again
		
		% Compute the absolute IPs
		true_CurrDict_AbsIPs = abs(param.trueDictionary' * newDict);
		
		% Allocate space for the order transformation
		atomTrns = zeros(1 , nAtoms);
		
		% Run for as many iterations as atoms
		for cFound = 1 : nAtoms
			
			% Find the maximal IP
			[bestTrueAtomInd , bestEstAtomInd] = find(true_CurrDict_AbsIPs == max(true_CurrDict_AbsIPs(:)));
			bestEstAtomInd = bestEstAtomInd(1); bestTrueAtomInd = bestTrueAtomInd(1); % In case we found more than one
			
			% Log this transformation
			atomTrns(bestTrueAtomInd) = bestEstAtomInd;
			
			% Make sure the same atoms aren't selected again
			true_CurrDict_AbsIPs(bestTrueAtomInd , :) = 0;
			true_CurrDict_AbsIPs(: , bestEstAtomInd) = 0;
			
		end
		
		% Perform the re-ordering
		newDict = newDict(: , atomTrns);
		
		% In some cases, atoms have reverse polarities. 
		% So, we want to change the polarities of the estimated atoms to match
		% that of the true atoms
		afterTrnsIPs = sum(newDict .* param.trueDictionary);
		signFactor = sign(afterTrnsIPs); signFactor(signFactor == 0) = 1;
		newDict = newDict .* repmat(signFactor , size(newDict,1) , 1);
		
		% Using the IPs after the transformation to 
		% detect if the true atoms are recovered
		isAtomDetected = abs(afterTrnsIPs) >= param.atomRecoveredThresh;
		fprintf('Percentage of found atoms = %03.2f\n' , 100 * mean(isAtomDetected));	
				
	end
	
	
	% Update the dictionary list
	allDicts{itr + 1} = newDict;
	currDict = newDict;
	
	% Show the dictionary if required
	if (param.showDictionary > 0) && (mod(itr , param.showDictionary) == 0)
		if mod(itr , 5) == 1, figure; end; % Not open new figure every time
		Show_Dictionary(currDict);
		title(sprintf('Dictionary after iteration %d' , itr));
		drawnow;
	end
	
	drawnow;
end

%% Return the last dictionary
resDict = currDict;

%% Finished
return;