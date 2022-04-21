function [Z] = alternating_back_projection2(Z,X,Y,F,G)
%ALTERNATING_BACK_PROJECTION Summary of this function goes here
%   Detailed explanation goes here

dZ_f0 = 1e0; dZ_fd = 1e0;
pF = pinv(F); pGt = G*inv((G')*G); iter = 1; tol = 0.001;
str = ['Iteration ',num2str(0), ': dZ_fd ', num2str(dZ_fd)]; disp(str);
spacer = 5; tic
while abs(dZ_fd) > tol
    Z = Z + pF*(Y - F*Z);
    dZt = pGt*((X-Z*G)'); %dZ_ff = norm(dZt,'fro');
    %dZ_fd = (dZ_f0 - dZ_ff)/dZ_f0;
    Z = Z + dZt'; %dZ_f0 = dZ_ff;
    remainder = rem(iter,2*spacer);
    if remainder == 2*spacer-1
        dZ_f0 = norm(dZt,'fro');
    elseif remainder == 0
        dZ_ff = norm(dZt,'fro');
        dZ_fd = (dZ_f0 - dZ_ff)/dZ_f0;
        avgTime = toc/(2*spacer); tic
        str = ['Iteration ',num2str(iter),': dZ_fd ',num2str(dZ_fd),...
            '    ',num2str(avgTime),' sec/iteration']; disp(str);
    end
    iter = iter + 1; 
end
end


