function [x, Ax, iter, objTrack, ukTrack] = RS_PAM(x, Sadjf, operators, ProxR, objective, opts)
%RS_PAM Runs the Proximal Alternating Minimization algorithm PAM [1].
% 
% [1] H. Attouch et al. "Proximal alternating minimization and projection 
% methods for nonconvex problems: An approach based on the Kurdyka-Lojasiewicz 
% inequality." 
% Mathematics of Operations Research 35.2 (2010): 438-457.
%  
%    Input:
%      x [cell]
%        initial guess for image and kernel
%      Sadjf [matrix]
%        upsampled data Sadj(f)
%      operators [struct]
%      	 contains several operators
%  
%  	 operators.A [function handle]
%  	   forward operator
%  	 operators.F [cell]
%          contains function handles for Fourier transformation
%  	 operators.C [function handle]
%  	   convolution operator operator 
%  	 operators.J [function handle]
%  	   embedding operator 
%  	 operators.Jadj [function handle]
%  	   adjoint embedding operator 
%  	 operators.SadjSInv [function handle]
%  	   inverse operator of (Sadj o S + rho * I)
%      ProxR [cell]
%         contains the prox operators for image and kernel update  
%      objective [struct]
%         contains function handles for evaluation of the objective function, the data 
%         fidelity and the regularizers
%      opts [struct]
%         various algorithmical parameters   	  
%  
%    Output:
%      x [cell]
%        optimal image and kernel
%      objTrack [matrix]
%         information about objective function values, number of ffts, time, etc.
%      ukTrack [cell]
%         various outputs for visualization and comparisons including
%         iterates of image and kernel
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
     
    A = operators.A; 
    F = operators.F;
    C = operators.C;
    J = operators.J;
    Jadj = operators.Jadj;
    SadjSInv = operators.SadjSInv;
    
    % initialize
    xm = x;
    imageSize = size(x{1});
    kernelSize = size(x{2});
   
    Fx = {F{1}(x{1}), F{2}(x{2})};
  
    Ax = A(Fx{2},Fx{1});
        
    
    
    initProxU.nu = zeros(imageSize);
    initProxU.mu = zeros(imageSize);    
    initProxU.z = zeros(imageSize);
    initProxU.p = zeros([imageSize 2]);   
    
    initProxK.nu = zeros(kernelSize);
    initProxK.mu = zeros(imageSize);    
    initProxK.xi = zeros(imageSize);       
    initProxK.z = x{2}; 
    initProxK.y = J(initProxK.z);
    initProxK.Fy = F{1}(initProxK.y);
    initProxK.p = zeros([kernelSize 2]);
    
    % options for ProxK and ProxU
    optsProxU.tol = opts.tolProxU;    
    optsProxU.minIter = opts.minIterProxU;
    optsProxU.maxIter = opts.maxIterProxU;    
    optsProxU.t = opts.tu;
    
    optsProxK.tol = opts.tolProxK;    
    optsProxK.minIter = opts.minIterProxK;
    optsProxK.maxIter = opts.maxIterProxK;
    optsProxK.t = opts.tk;
    
    rho_k = opts.rho_k; %rho_k remains fixed!!!!
    rho_u = 1;
        
    [objTrack,ukTrack] = RS_track_and_display_progress([],{},x{1},x{2},objective,opts.tracking_freq,opts.verbose_freq,opts.draw_iterates,opts.draw_iterates_fig,opts.niter,0,Ax);        
       
    for iter = 1 : opts.niter
        
        %% update image u
        [x{1},initProxU,rho_u] = RS_prox_PAM_u(x{1},Fx{2},Sadjf,operators,ProxR{1},optsProxU,initProxU,rho_u);
        Fx{1} = F{1}(x{1});
        
        %% update kernel k      
        [x{2}, initProxK,rho_k] = RS_prox_PAM_k(x{2},Fx{1},Sadjf,operators,ProxR{2},optsProxK,initProxK,rho_k);
        Fx{2} = F{2}(x{2});          
        
        Ax = A(Fx{2},Fx{1});
        
        %% track and display progress
        [objTrack,ukTrack] = RS_track_and_display_progress(objTrack,ukTrack,x{1},x{2},objective,opts.tracking_freq,opts.verbose_freq,opts.draw_iterates,opts.draw_iterates_fig,opts.niter,iter,Ax);

        %% check tolerance      
        diffu = norm(Column(x{1} - xm{1})) / norm(Column(x{1}));
        
        if (opts.tol > 0) && (diffu  < opts.tol)
            fprintf('(RSPam) Iteration %4d: Tolerance reached: difference between image iterates %2.2e < %2.2e\n', iter , diffu, tol);
            [objTrack,ukTrack] = RS_track_and_display_progress( ...
                objTrack, ukTrack, x{1},x{2},objective,opts.tracking_freq, ...
                opts.verbose_freq,opts.draw_iterates,opts.draw_iterates_fig, ...
                opts.niter, opts.niter, Ax);
            break;
        end
        xm = x;
    end
end
