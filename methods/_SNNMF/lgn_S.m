function [val] = lgn_S(B,z,y,x,s,h1,h2,lambda,mu)
% Lagrangian for ADMM_L
val = (1/2)*norm(z-B*y,2)^2 + lambda*norm(s,1) + (h1')*(x-s) + (h2')*(x-y)...
    + (mu/2)*norm(x-s,2)^2 + (mu/2)*norm(x-y,2)^2;
end