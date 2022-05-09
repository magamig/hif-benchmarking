function [k, init, rho] = ...
    RS_prox_PAM_k(kj, Fu,Sadjf, operators, ProxRk, opts, init, rho)
%RS_prox_PAM_k computes the proximal step for the kernel k
%inside one iteration of the PAM algorithm (second step of algorithm (3) in [1]).
%The solution is obtained by recasting the problem as a constraint
%minimizaiton problem and applying ADMM.
%
% [1] H. Attouch et al. "Proximal alternating minimization and projection 
% methods for nonconvex problems: An approach based on the Kurdyka-Lojasiewicz 
% inequality." 
% Mathematics of Operations Research 35.2 (2010): 438-457.
%  
%    Input:
%      kj [matrix]
%        current kernel k
%      Fu [matrix]
%        the image u in Fourier space
%      Sadjf [matrix]
%        upsampled data Sadj(f)
%      operators [struct]
%      	 contains several operators
%        operators.J [function handle]
%           embedding operator  
%        operators.Jadj [function handle]
%           adjoint embedding operator  
%        operators.C [function handle]
%           convolution operator  
%        operators.SadjSInv [function handle]
%           inverse of (Sadj o S + rho * I)
%      ProxRk [cell]
%         contains the prox operator for the kernel update
%      opts [struct]
%         various algorithmical parameters
%      init [struct]
%         contains several initial guesses
%         nu [matrix]
%           initial guess for the Lagrange multiplier nu
%         mu [matrix]
%           initial guess for the Lagrange multiplier mu
%         xi [matrix]
%           initial guess for the Lagrange multiplier xi
%         y [matrix]
%           initial guess for the variable y
%         Fy [matrix]
%           initial guess for the variable y in Fourier space
%         p [matrix]
%           initial guess for the dual state in ProxRk
%      rho [real]
%         initial step size for ADMM
%  
%    Output:
%      u [matrix]
%        new image
%      init [struct]
%         updated initial values
%      rho [real]
%         updated rho
%      
%     
% See also:
%
% -------------------------------------------------------------------------
% Copyright 2017, L. Bungert, D. Coomes, M. J. Ehrhardt, M. A. Gilles, 
% J. Rasch, R. Reisenhofer, C.-B. Schoenlieb
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------  

    nu = init.nu;
    mu = init.mu;
    xi = init.xi;
    y = init.y;
    Fy = init.Fy;
    p = init.p;
    
    SadjSInv = operators.SadjSInv;
    C = operators.C;
    J = operators.J;
    Jadj = operators.Jadj;
    
    
    Cuy = C(Fu,Fy);
    Cuy_old = Cuy;
    kold = kj;
    k = kj;
    y_old = y;
    
    
    conjFu = conj(Fu);
    sqAbsFu_plus1 = abs(Fu).^2 + 1;
    
    for n = 1 : opts.maxIter
        % First block
        % update x
        x = (SadjSInv(Sadjf + rho*(Cuy - mu), rho));
        Fxmu = RS_fourier(mu + x);
        % update z
        z = (opts.t*kj + rho*(Jadj(y - xi) + k - nu))/(opts.t + 2*rho);
        
        % Second block
        % update y
        Jz = J(z);
        Fy = conjFu.*Fxmu + RS_fourier(Jz + xi);
        Fy = Fy./sqAbsFu_plus1; % this may not be the Fourier transform of y!!!
        y = real(RS_fourier_inverse(Fy));
        % update k
        [k, p] = ProxRk(rho, z + nu, p);
             
        if n == opts.maxIter
            break
        end
        
        % update multipliers
        Cuy = C(Fu,Fy);
        
        mu = mu + rho*(x - Cuy);
        nu = nu + rho*(z - k);
        xi = xi + rho*(J(z) - y);
        
        % compute residuals
        prim_res = sqrt(sum((x(:) - Cuy(:)).^2) + sum((Jz(:) - y(:)).^2) + sum((z(:) - k(:)).^2));
        prim_res = prim_res / sqrt(numel(k) + 2*numel(x));

        Jadj_diff_y = Jadj(y - y_old);
        dk = k - kold;
        dual_res = rho * sqrt(sum((Cuy(:) - Cuy_old(:)).^2) + sum(Jadj_diff_y(:).^2 + dk(:).^2));
        dual_res = dual_res / sqrt(numel(k) + numel(x));
%         fprintf('(RSProxPamK) Iteration %4d: %2.2e, %2.2e\n',n , prim_res, dual_res);

%         mu_rho = 10;
%         tau_rho = 2;
%         if prim_res > mu_rho * dual_res
%             rho = rho * tau_rho;
%         elseif dual_res > mu_rho * prim_res
%             rho = rho / tau_rho;
%         end

%         tol = 1e-3;
        if n >= opts.minIter && prim_res < opts.tol && dual_res < opts.tol
%             fprintf('(RSProxPamK) Iteration %4d: %2.2e, %2.2e, rho:%2.2e\n',n , prim_res, dual_res, rho);       
            break
        end
                
        Cuy_old = Cuy;
        y_old = y;
        
%         % update dk and check tolerance
%         dk = kold - k;
%         diffk = norm(Column(dk)) / norm(Column(k));
% %         fprintf('(RSProxPamK) Iteration %4d: %2.2e\n',n , diffk);
%         if (opts.tol > 0) && (diffk  < opts.tol)
%             %fprintf('(RSProxPamK) Iteration %4d: Tolerance reached: difference between kernel iterates %2.2e < %2.2e\n',n , diffu, tol);
%             break;
%         end
        kold = k;
    end
    
    init.nu = nu;
    init.mu = mu;
    init.xi = xi;
    init.y = y;
    init.Fy = Fy;
    init.p = p;
    
end