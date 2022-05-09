function [u,init,rho] = RS_prox_PAM_u(uj, Fk, Sadjf, operators, ProxRu,opts,init,rho)
%RS_prox_PAM_u computes the proximal step for the image u
%inside one iteration of the PAM algorithm (first step of algorithm (3) in [1]).
%The solution is obtained by recasting the problem as a constraint
%minimizaiton problem and applying ADMM.
%
% [1] H. Attouch et al. "Proximal alternating minimization and projection 
% methods for nonconvex problems: An approach based on the Kurdyka-Lojasiewicz 
% inequality." 
% Mathematics of Operations Research 35.2 (2010): 438-457.
%  
%    Input:
%      uj [matrix]
%        current image u
%      Fk [matrix]
%        the kernel k in Fourier space
%      Sadjf [matrix]
%        upsampled data Sadj(f)
%      operators [struct]
%      	 contains several operators
%        operators.C [function handle]
%           convolution operator 
%        operators.SadjSInv [function handle]
%           inverse of (Sadj o S + rho * I)
%      ProxRu [cell]
%         contains the prox operator for the image update
%      opts [struct]
%         various algorithmical parameters
%      init [struct]
%         contains several initial guesses
%         nu [matrix]
%           initial guess for the Lagrange multiplier nu
%         mu [matrix]
%           initial guess for the Lagrange multiplier mu
%         z [matrix]
%           initial guess for the variable z
%         p [matrix]
%           initial guess for the dual state in ProxRu
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
    z = init.z;
    p = init.p;

    SadjSInv = operators.SadjSInv;
    C  = operators.C;
    
    sqAbsFk = abs(Fk).^2;
        
    Fz = RS_fourier(z);
        
    Ckz = C(Fk,Fz);
    Ckz_old = Ckz;
    
    z_old = z;
    for n = 1 : opts.maxIter
        % First block
        % update u
        [u, p] = ProxRu(rho, z - nu, p);
        % update x
        x = SadjSInv(Sadjf + rho*(Ckz - mu), rho);
        Fxmu = RS_fourier(rho*(mu + x));
        
        % Second block
        % update z
        Fz = conj(Fk).*Fxmu + RS_fourier(rho*(u + nu) + opts.t*uj);
        Fz = Fz./(rho*sqAbsFk + rho + opts.t);
        z = real(RS_fourier_inverse(Fz));
       
        % update multipliers
        Ckz = C(Fk,Fz);
        
        mu = mu + rho*(x - Ckz);
        nu = nu + rho*(u - z);
        
        % compute residuals
        prim_res = sqrt(sum((x(:) - Ckz(:)).^2) + sum((u(:) - z(:)).^2));
        prim_res = prim_res / sqrt(2*numel(x));

        dual_res = rho * sqrt(sum((Ckz(:) - Ckz_old(:)).^2) + sum((z(:) - z_old(:)).^2));
        dual_res = dual_res / sqrt(2*numel(x));
%         fprintf('(RSProxPamU) Iteration %4d: %2.2e, %2.2e\n',n , prim_res, dual_res);
        
        mu_rho = 10;
        tau_rho = 2;
        if prim_res > mu_rho * dual_res
            rho = rho * tau_rho;
        elseif dual_res > mu_rho * prim_res
            rho = rho / tau_rho;
        end

        if n >= opts.minIter && prim_res < opts.tol && dual_res < opts.tol
%             fprintf('(RSProxPamU) Iteration %4d: %2.2e, %2.2e, rho:%2.2e\n',n , prim_res, dual_res, rho);
            break
        end
        
        Ckz_old = Ckz;
        z_old = z;
        
%         % update du and check tolerance
%         du = uold - u;
%         diffu = norm(Column(du)) / norm(Column(u));
% %         fprintf('(RSProxPamU) Iteration %4d: %2.2e\n',n , diffu);
%         if (opts.tol > 0) && (diffu  < opts.tol)
%             %fprintf('(RSProxPamU) Iteration %4d: Tolerance reached: difference between image iterates %2.2e < %2.2e\n',n , diffu, tol);
%             break;
%         end
%         uold = u;
    end
    init.nu = nu;
    init.mu = mu;
    init.z = z;
    init.p = p;
end