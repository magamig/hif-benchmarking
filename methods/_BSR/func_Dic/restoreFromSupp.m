function xhat = restoreFromSupp(y, varargin)
%        compCode(y, D, subMean, maxAtoms, delta)
%
% dictionary based denoising
%   
%  -- input parameters 
%
%  y ->  degraded image
%
%  D -->  dictionary or support
%
%  subMean -> {0,1} subtract the mean of the patches
%
%  supp ->  active support for eact pactch of y
%
%
%  -- output arguments
%
%   
%   xhat ->   L(proj(y on the D_supp))    %  L is the patch averaging operator
%

% m number of pixels per patch
% k is the number of actoms of the dictionary
m = size(varargin{1},1);%m=m/size(y,3);
% take patches of y
patsize = sqrt(m);
% if strcmp(method,'BbB1')
%     for i=1:size(y,3)
%         Py_bb = im2col(y(:,:,i),[patsize patsize],'sliding');
%         Py=cat(1,Py,Py_bb);
%     end
% else

% % end
% if subMean == 1
%      meancolx = mean(Py);
%      Py=Py-repmat(meancolx,[size(Py,1) 1]);
% end
Py = im2col(y,[patsize patsize],'sliding');
[~, numpatches] = size(Py);
Py_hat = zeros(size(Py));

if length(varargin)==1 %% The input is the dictionary set
    D_S=varargin{1};
    for i=1:numpatches
        Py_hat(:,i) = D_S(:,:,i)*Py(:,i);
    end
else         %% The input is the support
    D=varargin{1}; 
    supp=varargin{2};
    for i=1:numpatches
        Daux =  D(:,supp(:,i));
        Py_hat(:,i) = Daux*(Daux\Py(:,i));
    end
end
%add the mean 
% if subMean == 1
%      Py_hat = Py_hat + repmat(meancolx,[size(Py_hat,1) 1]);
% end

blockx = patsize;blocky = patsize;

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