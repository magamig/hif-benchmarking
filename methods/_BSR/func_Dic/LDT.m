function code = LDT(y, D, supp, accum_estimates)
%        code = LDT(y, D, supp, accum_estimates)
%
% Operatotor code = transpose(LD)(y); 
%   
%  -- input parameters 
%
%  y ->   image
%
%  D -->  dictionary
%
%  supp ->  active support for eact pactch of y
%
%  accum_estimates  -->  number of times a pixel is in the set of patches
%
%  -- output arguments
%
%   
%   code ->   
%

% m number of pixels per patch
% k is the number of actoms of the dictionary
[m,k] = size(D);
% take patches of y
patsize = sqrt(m);

numAtoms = sum(supp(:,1) ~=0);

code = zeros(numAtoms,length(supp));

Py = im2col(y./accum_estimates,[patsize patsize],'sliding');

[num_atoms, numpatches] = size(code);

for i=1:numpatches
     Daux =  D(:,supp(:,i));
     code(:,i) = Daux'*Py(:,i);
end










