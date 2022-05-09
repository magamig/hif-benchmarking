function [xhat,alpha] = compCode(y, D, subMean, maxAtoms, delta,method)
%        compCode(y, D, subMean, maxAtoms, delta)
%
% dictionary based denoising
%   
%  -- input parameters 
%
%  y ->  degraded image
%
%  D -->  dictionary
%
%  subMean -> {0,1} subtract the mean of the patches
%
%  maxAtoms ->  is the maximum number of atoms in the sparse  regression of
%               each patch
%
%  delta    ->  is the maximum  error per pixel
%
%  -- output arguments
%
%   code ->   code for y
%   
%   xhat ->   L(D*alpha)    %  L is the patch averaging operator
%

% m number of pixels per patch
% k is the number of actoms of the dictionary
[m,k] = size(D);
% take patches of y
patsize = sqrt(m);
if strcmp(method,'Cube')
    Py = y;
else
    Py = im2col(y,[patsize patsize],'sliding');
end

if subMean == 1
     meancolx = mean(Py);
     Py=Py-repmat(meancolx,[size(Py,1) 1]);
end

% coding step
paramOMP.L = maxAtoms;
paramOMP.err = delta;
alpha=OMP_C(D,Py,paramOMP);

Py_hat = D*alpha;

% xhat =[];

%add the mean 
if subMean == 1
     Py_hat = Py_hat + repmat(meancolx,[size(Py_hat,1) 1]);
end

% if strcmp(method,'Cube')
%     Py_hat=reshape(Py_hat,nb_sub,size(Py_hat,1)/nb_sub,size(Py_hat,2));
%     Py_hat=shiftdim(Py_hat,1);
% end

% for i=1:size(Py_hat,3)
%     xhat=rebulid(Py_hat,patsize,)
% end

blockx = patsize;
blocky = patsize;
final_numestimate = zeros(size(y));
final_extentestimate = zeros(size(y));
for indexi = 1:blocky
    for indexj = 1:blockx
        tempesti = reshape(Py_hat((indexi-1)*blockx+indexj,:),size(y)-[blockx,blocky]+1);
        numestimate = zeros(size(y));
        extentestimate = zeros(size(y));
        extentestimate(1:size(tempesti,1),1:size(tempesti,2)) = tempesti;
        numestimate(1:size(tempesti,1),1:size(tempesti,2)) = 1;
        
        extentestimate = circshift(extentestimate, [indexj,indexi]-1);
        numestimate = circshift(numestimate, [indexj,indexi]-1);
        
        final_numestimate = final_numestimate+numestimate;
        final_extentestimate = final_extentestimate+extentestimate;
    end
end
xhat = final_extentestimate./final_numestimate;