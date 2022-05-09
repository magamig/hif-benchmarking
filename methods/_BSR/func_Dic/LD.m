function [z,final_numestimate] = LD(code, D, supp,size_im)
%       xhat = LD(cod, D, supp);
%
% Operatotor z = L(D(code)); restore an image from the code restricted to
%                            to supp
%   
%  -- input parameters 
%
%  code  -->  code for each patch with lenght sum(supp) 
%
%
%  D     -->  dictionary
%
%  supp   -->  active support for eact pactch of y
%
%
%  -- output arguments
%
%   
%   z ->   L(D(code)  %  L is the patch composing and averaging operator
%
%   accumulate the number of a pixel is in the set of patches
%

% m number of pixels per patch
% k is the number of actoms of the dictionary
[m,k] = size(D);
% take patches of y
patsize = sqrt(m);


[num_atoms, numpatches] = size(code);
Py = zeros(m,numpatches);

for i=1:numpatches
     Daux =  D(:,supp(:,i));
     Py(:,i) = Daux*code(:,i);
end

% patch composition

blockx = patsize;
blocky = patsize;
final_numestimate = zeros(size_im);
final_extentestimate = zeros(size_im);
for indexi = 1:blocky
    for indexj = 1:blockx
        tempesti = reshape(Py((indexi-1)*blockx+indexj,:),size_im-[blockx,blocky]+1);
        numestimate = zeros(size_im);
        extentestimate = zeros(size_im);
        extentestimate(1:size(tempesti,1),1:size(tempesti,2)) = tempesti;
        numestimate(1:size(tempesti,1),1:size(tempesti,2)) = 1;
        
        extentestimate = circshift(extentestimate, [indexj,indexi]-1);
        numestimate = circshift(numestimate, [indexj,indexi]-1);
        
        final_numestimate = final_numestimate+numestimate;
        final_extentestimate = final_extentestimate+extentestimate;
    end
end
z = final_extentestimate./final_numestimate;











