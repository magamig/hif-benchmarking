function Gamma = sparsecode(data,D,XtX,G,thresh)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             sparsecode               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


global CODE_SPARSITY codemode
global MEM_HIGH memusage
global ompparams


if (memusage < MEM_HIGH)
  Gamma = omp(D,data,G,thresh,ompparams{:});
  
else  % memusage is high
  
  if (codemode == CODE_SPARSITY)
    Gamma = omp(D'*data,G,thresh,ompparams{:});
    
  else
    Gamma = omp2(D'*data,XtX,G,thresh,ompparams{:});
  end
  
end

end
