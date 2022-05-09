function [val] = lgn_A(C,y,f,rho,alpha,mu)
% Lagrangian for ADMM_A
val = norm(f - C*alpha,2)^2 + (y')*(rho - alpha) + (mu/2)*norm(rho - alpha,2)^2;
end