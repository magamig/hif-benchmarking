function [Z] = alternating_back_projection(Z,X,Y,F,G,par,S)
%ALTERNATING_BACK_PROJECTION Summary of this function goes here
%   Detailed explanation goes here

dZ_f0 = 1e0; dZ_fd = 1e0;
pF = pinv(F);   GtG = inv((G')*G);   % 
pGt = G*inv((G')*G);
iter = 1; tol = 0.001;
%str = [num2str(iter), ': dZ_fd ', num2str(dZ_fd)]; disp(str);
while (abs(dZ_fd) > tol)&&(iter<200) 
    %tic
%        getrmse(hyperConvert3D(Z,256,256,93),S)
    Z = Z + pF*(Y - F*Z);
%     dZt =  (par.HT( (X-par.H(Z))*GtG' ))';
%  dZt =  (par.HT( (X-par.H(Z)) ))'; 
    dZt = pGt*((X-Z*G)'); 
    dZ_ff = norm(dZt,'fro');
    dZ_fd = (dZ_f0 - dZ_ff)/dZ_f0;
    Z = Z + dZt'; dZ_f0 = dZ_ff;
   iter = iter + 1; str = [num2str(iter), ': dZ_fd ', num2str(dZ_fd)]; disp(str);
    %toc
  
end
end



