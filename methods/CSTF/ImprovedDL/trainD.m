function [D,Coeffs,err] = trainD(D, data, Coeffs, TestData, params)
%%  Train dictionary
% =========================================================================
% INPUT ARGUMENTS:
% D                            The initial dictionary of size n X dictsize.
% data                         An nXN matrix that contains N signals (Y), each of dimension n. 
% Coeffs (optional)            Initial coefficients, if any, of size
%                                   dictsize X N. 
% TestData (optional)          An nXN' matrix that contains N' different signals (Y), 
%                                   each of dimension n, for testing the new D. 
% params                        structure that includes all required
%                                 parameters for the K-SVD execution.
%    Tdata, ...                The sparisty constraint
%    Edata, ...                The error constraint
%    dictsize,...              Number of atoms in the dictionary.
%    iternum,...               Number of iterations to perform.
%    DUCs, ...                 Number of dictionary update cycles per iteration (default=1)
%    MOD_LSX, ...              Use MOD_LS dictionary update method (default=false)
%    codemode                  Specifies constraint of either 'sparsity' or 'error' (default='sparsity')
%    epsilon                   Specifies error with 'error' constraint (default=Edata) 
%    sparseMeth                Sparse coding method; 'OMP', 'MatlabOMP', 'CoefROMP', 'batchCoefROMP'  (default='OMP'))
%    	If sparse coding method is 'CoefROMP' or 'batchCoefROMP':  
%    addK                      Number of coefficients to add each iteration (default=n/30)
%    card                      Cardinality with the 'sparsity' constraint (default=n/10)
%    addX                      Cardinality with the 'sparsity' constraint (default=2)
%    maxAtoms                  Maximum cardinality allowed (default=n/4)
%    startRepl                 Iteration to start using CoefROMP (default=1)
% =========================================================================
% OUTPUT ARGUMENTS:
%  Dictionary                  The extracted dictionary of size nX(param.K).
%  Coeffs                      Final coefficients of size dictsize X N. 
%  err                         RMSE for each iteration.  If TestData is
%                                provided, 2 vectors of RMSE; one for the
%                                data and the other for the TestData
% =========================================================================
global CODE_SPARSITY CODE_ERROR codemode
global ompparams exactsvd

%   Initialize
CODE_SPARSITY = 1;
CODE_ERROR = 2;

if (isfield(params,'codemode'))
    switch lower(params.codemode)
        case 'sparsity'
            codemode = CODE_SPARSITY;
            thresh = params.Tdata;
        case 'error'
            codemode = CODE_ERROR;
            thresh = params.Edata;
        otherwise
            error('Invalid coding mode specified');
    end
elseif (isfield(params,'Tdata'))
    codemode = CODE_SPARSITY;
    thresh = params.Tdata;
elseif (isfield(params,'Edata'))
    codemode = CODE_ERROR;
    thresh = params.Edata;
else
    error('Data sparse-coding target not specified');
end
%     thresh = blksz * params.sigma * params.gain;   % target error for omp
%     fprintf(' Error threshold = %g  \n',thresh);

ompparams = {'checkdict','off'};
ompparams{end+1} = 'messages';
ompparams{end+1} = 0;

    function out = setParams( field, default )
        if ~isfield( params, field )
            params.(field)    = default;
        end
        out = params.(field);
    end

iternum = setParams( 'iternum', 2 );
DUCs = setParams( 'DUC', 1 );
dim = size(D,1);
nSignals = size(data,2);
Tdata  = setParams( 'Tdata', 12 );
startRepl= setParams( 'startRepl', 1 );
% sim= setParams( 'sim', [] );
% testSim= setParams( 'testSim', [] );
sparseMeth =  setParams( 'sparseMeth', 'OMP' );

%   Initialize parameters for CoROMP
params.addK  = setParams( 'addK', round(dim/30) );
params.card  = setParams( 'card', round(dim/10) );
params.maxAtoms  = setParams( 'maxAtoms', round(dim/4) );
addX = setParams( 'addX', 2 );
MOD_LSX = setParams( 'MOD_LSX', 0 );
muthresh = 0.99;

if codemode == CODE_ERROR
    ompparams{end+1} = 'maxatoms';
    ompparams{end+1} = params.maxAtoms;
end

isTestData =  ~isempty(TestData);
params.epsilon = setParams( 'epsilon', thresh );
params.coeffThres  = setParams( 'coeffThres', thresh/4 );
XtX = colnorms_squared(data);
if isTestData
    testXtX = colnorms_squared(TestData);
end

% determine dictionary size %

if (any(size(D)==1) && all(iswhole(D(:))))
    dictsize = length(D);
else
    dictsize = size(D,2);
end
if (isfield(params,'dictsize'))    % this superceedes the size determined by initdict
    dictsize = params.dictsize;
end

if (nSignals < dictsize)
    error('Number of training signals is smaller than number of atoms to train');
end


% initialize the dictionary %
if (any(size(D)==1) && all(iswhole(D(:))))
    D = data(:,D(1:dictsize));
else
    if (size(D,1)~=size(data,1) || size(D,2)<dictsize)
        error('Invalid initial dictionary');
    end
    D = D(:,1:dictsize);
end
% normalize the dictionary %
D = D*diag(1./sqrt(sum(D.*D)));

replaced_atoms = zeros(1,dictsize);  % mark each atom replaced by optimize_atom
unused_sigs = 1:nSignals;  % tracks the signals that were used to replace "dead" atoms.
% makes sure the same signal is not selected twice

fprintf('trainD: sparseMeth,dictsize,Tdata,thresh,maxAtoms,MOD_LSX= %s, %g, %g, %g, %g, %g \n', ...
    sparseMeth,dictsize,Tdata,thresh,params.maxAtoms,MOD_LSX);

%   main loop through iterations
TestCoeffs = [];
err = zeros(iternum,2);
for iter = 1:iternum
    %   Sparse coding
    params.addX = addX/iter;
    
    if strcmp(sparseMeth,'CoefROMP') && iter >= startRepl
        fprintf('trainD: addK,addX,card,DUCs,epsilon,coeffThres,startRepl= %g, %g, %g, %g, %g, %g, %g \n', ...
            params.addK,params.addX,params.card,DUCs,params.epsilon,params.coeffThres,startRepl);
%         params.sim = sim;
        [~, Coeffs] = CoROMP3(D , data , Coeffs , params);
    elseif strcmp(sparseMeth,'batchCoefROMP')  && iter >= startRepl
        fprintf('trainD: addK,addX,card,DUCs,epsilon,coeffThres,startRepl= %g, %g, %g, %g, %g, %g, %g \n', ...
            params.addK,params.addX,params.card,DUCs,params.epsilon,params.coeffThres,startRepl);
        G = D'*D;
        [~, Coeffs] = Batch_CoefROMP(D , data , Coeffs , G , params);
    elseif strcmp(sparseMeth,'MatlabOMP')
        [D, Coeffs] = MatlabOMP(D , data , sumCoeffs , params);
    else
        G = D'*D;
        Coeffs = sparsecode(data,D,XtX,G,thresh);
    end
    
    tst = zeros(1,dictsize);
    for i1 = 1:dictsize
        tst(i1) = nnz(Coeffs(i1,:));
    end
    if find(tst==0)
        disp(['trainD: Unused atoms ' num2str(find(tst==0)) ])
    end
    
    %   dictionary update
    for duc = 1:DUCs
        if MOD_LSX == 1
            D=data*Coeffs'/(Coeffs*Coeffs');
            Scale=diag(1./sqrt(diag(D'*D)));
            D=D*Scale;
            
            % Update coefficients Coeffs
            for k=1:1:nSignals
                support=find(Coeffs(:,k)~=0);
                Coeffs(support,k)=D(:,support)\data(:,k);
            end;
        else    % K-SVD
            p = 1:dictsize; % randperm(dictsize);  % 
            for j = 1:dictsize
                [D(:,p(j)),gamma_j,data_indices,unused_sigs,replaced_atoms] = optimize_atom(data,D,p(j),Coeffs,unused_sigs,replaced_atoms);
                Coeffs(p(j),data_indices) = gamma_j;
            end
        end
    end
    if isTestData
        if strcmp(sparseMeth,'CoefROMP')  && iter >= startRepl
%             params.sim = testSim;
            [~, TestCoeffs] = CoROMP3(D , TestData , TestCoeffs , params);
        elseif strcmp(sparseMeth,'batchCoefROMP') && iter >= startRepl
            G = D'*D;
            [~, TestCoeffs] = Batch_CoefROMP(D , TestData , TestCoeffs , G , params);
        else
            G = D'*D;
            TestCoeffs = sparsecode(TestData,D,testXtX,G,Tdata);
        end
        err(iter,2) = compute_err(D,TestCoeffs,TestData);
        err(iter,1) = compute_err(D,Coeffs,data);
        fprintf(' Iteration %g / %g  RMSE = %g,  testdata RMSE = %g, ',iter,iternum,err(iter,1),err(iter,2));
    else
        err(iter,1) = compute_err(D,Coeffs,data);
        fprintf(' Iteration %g / %g  RMSE = %g, ',iter,iternum,err(iter,1));
    end
    avgSparsity = nnz(Coeffs) / size(Coeffs,2);
    fprintf(' average sparsity = %g ',avgSparsity);
    
    [D,cleared_atoms] = cleardict(D,Coeffs,data,muthresh,unused_sigs,replaced_atoms);
    fprintf(', replaced %d atoms  \n', sum(replaced_atoms)+cleared_atoms);

end

avgSparsity = nnz(Coeffs) / size(Coeffs,2);
fprintf(' Final average sparsity = %g  ',avgSparsity);
fprintf(' Final  RMSE = %g  \n',compute_err(D,Coeffs,data));
return
end


function [atom,gamma_j,data_indices,unused_sigs,replaced_atoms] = optimize_atom(X,D,j,Gamma,unused_sigs,replaced_atoms)

global exactsvd

% data samples which use the atom, and the corresponding nonzero
% coefficients in Gamma
% [gamma_j, data_indices] = sprow(Gamma, j);
data_indices = find(Gamma(j,:));

if (length(data_indices) < 1)
    maxsignals = 60000;
    perm = randperm(length(unused_sigs));   %   1:length(unused_sigs);   %   
    perm = perm(1:min(maxsignals,end));
    Err = X(:,unused_sigs(perm)) - D*Gamma(:,unused_sigs(perm));
    E = sum(Err.^2);
    [~,i] = max(E);
    atom = X(:,unused_sigs(perm(i)));
    atom = atom./norm(atom);
    atom = sign(atom(1)) * atom;
    gamma_j = zeros(length(data_indices),1);
    unused_sigs = unused_sigs([1:perm(i)-1,perm(i)+1:end]);
    replaced_atoms(j) = 1;
    return;
end

gamma_j = Gamma(j,data_indices);
smallGamma = Gamma(:,data_indices);
Dj = D(:,j);

if (exactsvd)
    [atom,s,gamma_j] = svds(X(:,data_indices) - D*smallGamma + Dj*gamma_j, 1);
    gamma_j = s*gamma_j;
else
    atom = X(:,data_indices)*gamma_j' - D*(smallGamma*gamma_j') + Dj*(gamma_j*gamma_j');
    atom = atom/norm(atom);
    gamma_j = atom'*X(1:size(X,1),data_indices) - (atom'*D)*smallGamma + (atom'*Dj)*gamma_j;
    
end

end

function [D,cleared_atoms] = cleardict(D,Gamma,X,muthresh,unused_sigs,replaced_atoms)

use_thresh = 4;  % at least this number of samples must use the atom to be kept

dictsize = size(D,2);

% compute error in blocks to conserve memory
err = zeros(1,size(X,2));
blocks = [1:3000:size(X,2) size(X,2)+1];
for i = 1:length(blocks)-1
    error(:,blocks(i):blocks(i+1)-1) = X(:,blocks(i):blocks(i+1)-1)-D*Gamma(:,blocks(i):blocks(i+1)-1);
    err(blocks(i):blocks(i+1)-1) = sum(error(:,blocks(i):blocks(i+1)-1).^2);
end

cleared_atoms = 0;
usecount = sum(abs(Gamma)>1e-7, 2);
for j = 1:dictsize
    % compute G(:,j)
    Gj = D'*D(:,j);
    Gj(j) = 0;
    
    % replace atom
    if ( (max(Gj.^2)>muthresh^2 || usecount(j)<use_thresh) && ~replaced_atoms(j) )
        [~,i] = max(err(unused_sigs));
%         D(:,j) = error(:,unused_sigs(i)) / norm(error(:,unused_sigs(i)));  %
        D(:,j) = X(:,unused_sigs(i)) / norm(X(:,unused_sigs(i)));  %
        unused_sigs = unused_sigs([1:i-1,i+1:end]);
        cleared_atoms = cleared_atoms+1;
    end
end

end



function Y = colnorms_squared(X)

% compute in blocks to conserve memory
Y = zeros(1,size(X,2));
blocksize = 2000;
for i = 1:blocksize:size(X,2)
    blockids = i : min(i+blocksize-1,size(X,2));
    Y(blockids) = sum(X(:,blockids).^2);
end

end

function err = compute_err(D,Gamma,data)
err = sqrt(sum(reperror2(data,D,Gamma))/numel(data));

end

function err2 = reperror2(X,D,Gamma)

% compute in blocks to conserve memory
err2 = zeros(1,size(X,2));
blocksize = 2000;
for i = 1:blocksize:size(X,2)
    blockids = i : min(i+blocksize-1,size(X,2));
    err2(blockids) = sum((X(:,blockids) - D*Gamma(:,blockids)).^2);
end

end
