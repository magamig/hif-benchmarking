function [Z] = alternating_back_projection_blur(Z,X,Y,F,G, psfY)
% Authors: Eliot Wycoff, T. Chan, et al.,
% E. Wycoff, T. Chan, K. Jia, W. Ma, and Y. Ma, "A non-negative sparse promoting algorithm for high
% resolution hyperspectral imaging,", ICASSP 2013.
% ALTERNATING_BACK_PROJECTION Summary of this function goes here
% Detailed explanation goes here
%   

dZ_f0 = 1e0; dZ_fd = 1e0;
pF = pinv(F);   GtG = inv((G')*G);   % 
% pGt = G*inv((G')*G);
iter = 1; tol = 0.001;
%str = [num2str(iter), ': dZ_fd ', num2str(dZ_fd)]; disp(str);
while (abs(dZ_fd) > tol)&&(iter<40)
 
    Z = Z + pF*(Y - F*Z);
   XXX1= X-creat_HSI(Z,psfY);
    XXX2=creat_HSI_T(XXX1*GtG,psfY);
  dZt = XXX2' ;
%     dZt =  (par.HT( (X-par.H(Z))' ))';   % 
%     dZt = pGt*((X-Z*G)'); 
    dZ_ff = norm(dZt,'fro');
    dZ_fd = (dZ_f0 - dZ_ff)/dZ_f0;
    Z = Z + dZt'; dZ_f0 = dZ_ff;
    iter = iter + 1; str = [num2str(iter), ': dZ_fd ', num2str(dZ_fd)]; disp(str);
    %toc
end
end

