function [IOut,output,CoefMatrix,blkMatrix] = denoiseImageKSVD(Image,sigma,K,initD,CoefMatrix,varargin)
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A  D I C T  I O N A R Y
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
% function IOut = denoiseImageKSVD(Image,sigma,K,varargin)
% denoise an image by sparsely representing each block with the
% already overcomplete trained Dictionary, and averaging the represented parts.
% Detailed description can be found in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
% This function may take some time to process. Possible factor that effect
% the processing time are:
%  1. number of KSVD iterations - the default number of iterations is 10.
%  However, fewer iterations may, in most cases, result an acceleration in
%  the process, without effecting  the result too much. Therefore, when
%  required, this parameter may be re-set.
%  2. maxBlocksToConsider - The maximal number of blocks to train on. If this 
%  number is larger the number of blocks in the image, random blocks
%  from the image will be selected for training. 
% ===================================================================
% INPUT ARGUMENTS : Image - the noisy image (gray-level scale)
%                   sigma - the s.d. of the noise (assume to be white Gaussian).
%                   K - the number of atoms in the trained dictionary.
%    Optional arguments:              
%                  'blockSize' - the size of the blocks the algorithm
%                       works. All blocks are squares, therefore the given
%                       parameter should be one number (width or height).
%                       Default value: 8.
%                  'errorFactor' - a factor that multiplies sigma in order
%                       to set the allowed representation error. In the
%                       experiments presented in the paper, it was set to 1.15
%                       (which is also the default  value here).
%                  'maxBlocksToConsider' - maximal number of blocks that
%                       can be processed. This number is dependent on the memory
%                       capabilities of the machine, and performances’
%                       considerations. If the number of available blocks in the
%                       image is larger than 'maxBlocksToConsider', the sliding
%                       distance between the blocks increases. The default value
%                       is: 250000.
%                  'slidingFactor' - the sliding distance between processed
%                       blocks. Default value is 1. However, if the image is
%                       large, this number increases automatically (because of
%                       memory requirements). Larger values result faster
%                       performances (because of fewer processed blocks).
%                  'numKSVDIters' - the number of KSVD iterations processed
%                       blocks from the noisy image. If the number of
%                       blocks in the image is larger than this number,
%                       random blocks from all available blocks will be
%                       selected. The default value for this parameter is:
%                       10 if sigma > 5, and 5 otherwise.
%                  'maxNumBlocksToTrainOn' - the maximal number of blocks
%                       to train on. The default value for this parameter is
%                       65000. However, it might not be enough for very large
%                       images
%                  'displayFlag' - if this flag is switched on,
%                       announcement after finishing each iteration will appear,
%                       as also a measure concerning the progress of the
%                       algorithm (the average number of required coefficients
%                       for representation). The default value is 1 (on).
%                  'waitBarOn' - can be set to either 1 or 0. If
%                       waitBarOn==1 a waitbar, presenting the progress of the
%                       algorithm will be displayed.
% OUTPUT ARGUMENTS : IOut - a 2-dimensional array in the same size of the
%                       input image, that contains the cleaned image.
%                    output.D - the trained dictionary.
% =========================================================================

% first, train a dictionary on the noisy image

p = ndims(Image);
% if (p<2 || p>3)
%   error('DENOISE only supports 2-D and 3-D signals.');
% else
%     disp([num2str(p) 'D problem'])
% end

reduceDC = 1;
[NN1,NN2,NN3] = size(Image);
waitBarOn = 1;
if (sigma > 5)
    numIterOfKsvd = 10;
else
    numIterOfKsvd = 5;
end
C = 1.15;
maxBlocksToConsider = 260000;
slidingDis = 1;
bb = 8;
maxNumBlocksToTrainOn = 65000;
displayFlag = 0;
DUC = 1;
image2Denoise = 137;
errorFlag = 1;
addK = 1;
addX = 0.9;
coeffFact = 0.8;

for argI = 1:2:length(varargin)
    if (strcmp(varargin{argI}, 'slidingFactor'))
        slidingDis = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'errorFactor'))
        C = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'maxBlocksToConsider'))
        maxBlocksToConsider = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'numKSVDIters'))
        numIterOfKsvd = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'blockSize'))
        bb = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'maxNumBlocksToTrainOn'))
        maxNumBlocksToTrainOn = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'displayFlag'))
        displayFlag = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'errorFlag'))
        errorFlag = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'waitBarOn'))
        waitBarOn = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'DUC'))
        DUC = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'addK'))
        addK = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'addX'))
        addX = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'coeffFact'))
        coeffFact = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'image2Denoise'))
        image2Denoise = varargin{argI+1};
    end
end

% if (sigma <= 5)
%     numIterOfKsvd = 5;
% end

% first, train a dictionary on blocks from the noisy image
if p == 2
    lenAtom = bb^2;
    if(prod([NN1,NN2]-bb+1)> maxNumBlocksToTrainOn)
        randPermutation =  randperm(prod([NN1,NN2]-bb+1));
        selectedBlocks = randPermutation(1:maxNumBlocksToTrainOn);
%         selectedBlocks = 1:maxNumBlocksToTrainOn;

        blkMatrix = zeros(bb^2,maxNumBlocksToTrainOn);
        for i = 1:maxNumBlocksToTrainOn
            [row,col] = ind2sub(size(Image)-bb+1,selectedBlocks(i));
            currBlock = Image(row:row+bb-1,col:col+bb-1);
            blkMatrix(:,i) = currBlock(:);
        end
    else
        blkMatrix = im2col(Image,[bb,bb],'sliding');
    end
elseif p == 3
    lenAtom = bb^3;
    numBlocks = min(prod([NN1,NN2,NN3]-bb+1), maxNumBlocksToTrainOn);
    randPermutation =  randperm(numBlocks);
    selectedBlocks = randPermutation(1:numBlocks);

    blkMatrix = zeros(lenAtom,numBlocks);
    for i = 1:numBlocks
        [row,col,z] = ind2sub(size(Image)-bb+1,selectedBlocks(i));
        currBlock = Image(row:row+bb-1,col:col+bb-1,z:z+bb-1);
        blkMatrix(:,i) = currBlock(:);
    end
end
param.K = K;
param.numIteration = numIterOfKsvd ;

param.errorFlag = errorFlag; % decompose signals until a certain error is reached. do not use fix number of coefficients.
param.errorGoal = sigma*C;
param.preserveDCAtom = 0;
param.L = round(bb^2/10);

if initD
    param.initialDictionary = initD;
else
    ln = ceil(sqrt(lenAtom));
    Pn=ceil(sqrt(K));
    DCT=zeros(ln,Pn);
    for k=0:1:Pn-1,
        V=cos([0:1:ln-1]'*k*pi/Pn);
        if k>0, V=V-mean(V); end;
        DCT(:,k+1)=V/norm(V);
    end;
    DCT=kron(DCT,DCT);
    param.initialDictionary = DCT(1:lenAtom,1:param.K );
end
param.InitializationMethod =  'GivenMatrix';

if (reduceDC)
    vecOfMeans = mean(blkMatrix);
    blkMatrix = blkMatrix-ones(size(blkMatrix,1),1)*vecOfMeans;
end

if (waitBarOn)
    counterForWaitBar = param.numIteration+1;
    h = waitbar(0,'Denoising In Process ...');
    param.waitBarHandle = h;
    param.counterForWaitBar = counterForWaitBar;
end

param.displayProgress = displayFlag;
param.DUC = DUC;
param.addK = addK;
param.addX = addX;
% param.incrAdd = 1;
param.coeffFact = coeffFact;
param.gain = C;
param.sigma = sigma;
param.maxAtoms  = round(bb^2/4);

[Dictionary,output,CoefMatrix] = KSVD2LNS(blkMatrix,CoefMatrix,param);
output.D = Dictionary;

errT = sigma*C;
if (displayFlag)
    disp('finished Trainning dictionary');
    fprintf('denoiseImageKSVD: sigma,errT,errorFactor,blockSize,maxNumBlocksToTrainOn,DUC,addK,addX,coeffFact= \n %g, %g, %g, %g, %g, %g, %g, %g, %g \n', ...
            sigma,errT,C,bb,maxNumBlocksToTrainOn,DUC,addK,addX,coeffFact);
end

% denoise the image using the resulted dictionary
while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end

if (waitBarOn)
    newCounterForWaitBar = (param.numIteration+1)*size(IOut,2)*size(IOut,3);
end

if p == 2
    IOut = zeros(size(Image));
    [blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);
    % go with jumps of 30000
    for jj = 1:30000:size(blocks,2)
        if (waitBarOn)
            waitbar(((param.numIteration*size(blocks,2))+jj)/newCounterForWaitBar);
        end
        jumpSize = min(jj+30000-1,size(blocks,2));
        if (reduceDC)
            vecOfMeans = mean(blocks(:,jj:jumpSize));
            blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
        end

        %Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT);
        Coefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),errT);
        if (reduceDC)
            blocks(:,jj:jumpSize)= Dictionary*Coefs + ones(size(blocks,1),1) * vecOfMeans;
        else
            blocks(:,jj:jumpSize)= Dictionary*Coefs ;
        end
    end

    count = 1;
    Weight = zeros(NN1,NN2);
    IMout = zeros(NN1,NN2);
    [rows,cols] = ind2sub(size(Image)-bb+1,idx);
    for i  = 1:length(cols)
        col = cols(i); row = rows(i);        
        block =reshape(blocks(:,count),[bb,bb]);
        IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
        Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
        count = count+1;
    end;

    IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);

elseif p == 3
    % the current batch of blocks
    blocksize = [bb bb bb];
    stepsize = [1 1 1];
    blk3sz = round(bb/2);
    sz = size(Image);
    sz(3) = 2*blocksize(3);
    IOut = zeros(sz);
    tic
%     for k = 1:stepsize(3):size(IOut,3)-blocksize(3)+1
    for k = image2Denoise-blocksize(3)+1:stepsize(3):image2Denoise
      for j = 1:stepsize(2):size(IOut,2)-blocksize(2)+1
        if (waitBarOn)
            waitbar(((param.numIteration*size(blocks,2))+j)/newCounterForWaitBar);
        end

        % the current batch of blocks
        blocks = im2colstep(Image(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1),blocksize,stepsize);
%         blocks = im2col(Image(:,j:j+blocksize(2)-1),[bb bb],'sliding');

        % remove DC
%         [blocks, dc] = remove_dc(blocks,'columns');
        dc = mean(blocks);
        blocks = bsxfun(@minus,blocks, dc);

        % denoise the blocks
        gamma = OMPerr(Dictionary,blocks,errT);
%         nz = nz + nnz(gamma);
        % add DC
        cleanblocks = bsxfun(@plus,Dictionary*gamma, dc);

        cleanvol = col2imstep(cleanblocks,[size(IOut,1) blocksize(2:3)],blocksize,stepsize);
%         cleanvol = col2imstep(cleanblocks,[size(IOut,1) blocksize(2)],blocksize(1:2),stepsize(1:2));
        k1 = k - image2Denoise + blocksize(3);
%         fprintf('  %g, %g, %g  \n',k,k1,k1+blocksize(3)-1);
        IOut(:,j:j+blocksize(2)-1,k1:k1+blocksize(3)-1) = IOut(:,j:j+blocksize(2)-1,k1:k1+blocksize(3)-1) + cleanvol;

      end
    end

    % average the denoised and noisy signals
%     cnt = countcover(size(Image),blocksize,stepsize);
    cnt = countcover(sz,blocksize,stepsize);
    toc
%     cnt = countcover(size(Image),blocksize,stepsize);
    % IOut = (IOut+lambda*Image)./(cnt + lambda);
    IOut = IOut./cnt;
end


if (waitBarOn)
    close(h);
end

