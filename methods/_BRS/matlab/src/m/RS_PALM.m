function [x, A_u_k, t, objTrack, ukTrack] = RS_PALM( ...
    xm, A, F, gradD, L, ProxR, objective, opts)
%RS_PALM Runs the Proximal Alternating Linearized Minimization algorithm PALM [1] 
% or its inertial variant iPALM [2].
% 
% [1] Bolte, Jerome, Shoham Sabach, and Marc Teboulle. "Proximal alternating 
% linearized minimization for nonconvex and nonsmooth problems." 
% Mathematical Programming 146.1-2 (2014): 459-494.
% 
% [2] Pock, Thomas, and Shoham Sabach. "Inertial Proximal Alternating 
% Linearized Minimization (iPALM) for nonconvex and nonsmooth problems." 
% SIAM Journal on Imaging Sciences 9.4 (2016): 1756-1787.
% 
% Input:
%     xm [cell] 
%         initial guess for image and kernel
%     A [function handle]
%         forward operator
%     F [cell]
%         contains function handles for Fourier transformation
%     gradD [cell]
%         contains function handles for the gradient of the differentiable part
%         D of the objective function
%     L [cell]
%         initial guess for Lipschitz constants of gradD{1} and gradD{2}
%     ProxR [cell]
%         contains the prox operators for image and kernel update
%     objective [struct]
%         contains function handles for evaluation of the objective function, the data 
%         fidelity and the regularizers
%     opts [struct]
%         various algorithmical parameters
%     
% Output:
%     x [cell]
%         optimal image and kernel
%     A_u_k [matrix]
%         application of the forward operator to optimal image and kernel
%     t [int]
%         number of PALM iterations
%     objTrack [matrix]
%         information about objective function values, number of ffts, 
%         time, etc.
%     ukTrack [cell]
%         various outputs for visualization and comparisons including
%         iterates of image and kernel
%     
% See also:
%
% -------------------------------------------------------------------------
% Copyright 2017, L. Bungert, D. Coomes, M. J. Ehrhardt, J. Rasch, 
% R. Reisenhofer, C.-B. Schoenlieb
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
      
        
    % choose inertia rule
    if ~opts.inertia
        alpha = @(t) 0;
    else
        if strcmp(opts.inertia_alpha, 'heuristic')
            alpha = @(t) (t-1)/(t+2);
        else
            alpha = @(t) opts.inertia_alpha;
        end
    end
    
    % choose step size rule
    if strcmp(opts.inertia_alpha, 'heuristic')
        tau = @(alpha, L, theta) theta * L;
    else
        tau = @(alpha, L, theta) theta * 0.5 * (1 + 2 * alpha) / (1 - alpha) * L;
    end
        
    inner = @(x, y) dot(x(:), y(:));
    sqNorm = @(x) sum(abs(x(:)).^2);
   
    p = {zeros([size(xm{1}), 2]), zeros([size(xm{2}), 2])};    
    xmm = xm;
    x = xm; % needed if kernel is not updated
    
    Fxm = {F{1}(xm{1}), F{2}(xm{2})};
    Fxmm = Fxm;
    Fx = Fxm; % needed if kernel is not updated
        
    A_um_km = A(Fxm{2}, Fxm{1});
        
    Ru_um = objective.image(x{1});
    Rk_km = objective.kernel(x{2});
    data_um_km = objective.data(A_um_km);
                
    [objTrack,ukTrack] = RS_track_and_display_progress([],{}, xm{1}, xm{2}, ...
        objective, opts.tracking_freq, ...
        opts.verbose_freq, opts.draw_iterates, ...
        opts.draw_iterates_fig, opts.niter,0, A_um_km, 0, 0, 0, 0);        
    
    for t = 1 : opts.niter
        % update image u
        if opts.inertia
            xt = (1 + alpha(t)) * xm{1} - alpha(t) * xmm{1};
            A_ut_km = (1 + alpha(t)) * A_um_km - alpha(t) * A(Fxm{2}, Fxmm{1});
            gu_ut_km = gradD{1}(Fxm{2}, A_ut_km);
            data_ut_km = objective.data(A_ut_km);
        else
            xt = xm{1};
            gu_ut_km = gradD{1}(Fxm{2}, A_um_km);
            data_ut_km = data_um_km;
        end
                
        backtrackingSuccess = false;
        for i_backtracking = 0 : 100    
            tau_ = tau(alpha(t), L{1}, opts.theta{1});
                        
            [x{1}, p{1}] = ProxR{1}(tau_, xt - gu_ut_km/tau_, p{1});
            Fx{1} = F{1}(x{1});
            A_u_km = A(Fxm{2}, Fx{1});
            
            data_u_km = objective.data(A_u_km);
                        
            LHS1 = data_u_km;
            RHS1 = data_ut_km + inner(gu_ut_km, x{1} - xt) ...
                + 0.5 * L{1} * sqNorm(x{1} - xt) ...
                + opts.ineq_tol;

            if LHS1 > RHS1 + opts.ineq_tol;
                L{1} = min(L{1} * opts.eta_up{1}, opts.L_max{1});
                continue;
            end
            
            Ru_u = objective.image(x{1});
                        
            % test 3.7 (without convexity)
            % in notation of iPALM:
            % sigma(u+) <= sigma(u) + <u - u+, grad h(w)> + t/2 ||u - v||^2 - t/2 ||u+ - v||^2
            LHS2 = Ru_u;
            RHS2 = Ru_um + inner(xm{1} - x{1}, gu_ut_km) ...
                + tau_/2 * (sqNorm(xm{1} - xt) - sqNorm(x{1} - xt)) ...
        		+ opts.ineq_tol;
            
            if LHS2 <= RHS2 + opts.ineq_tol;
                L{1} = max(L{1} / opts.eta_down{1}, opts.L_min{1});
                backtrackingSuccess = true;
                break
            end
        end
        if ~backtrackingSuccess
            x{1} = xm{1};
            Fx{1} = Fxm{1};
            A_u_km = A_um_km;
            warning(['No step size found for image in iteration ', num2str(t)])
        end
        
        %% update kernel k
        if opts.update_kernel
            if opts.inertia
                xt = (1 + alpha(t)) * xm{2} - alpha(t) * xmm{2};
                A_u_kt = (1 + alpha(t)) * A_u_km - alpha(t) * A(Fxmm{2}, Fx{1});
                gk_u_kt = gradD{2}(Fx{1}, A_u_kt);
                data_u_kt = objective.data(A_u_kt);
            else
                xt = xm{2};
                data_u_kt = data_u_km;
                gk_u_kt = gradD{2}(Fx{1}, A_u_km);
            end

            backtrackingSuccess = false;
            for i_backtracking = 0 : 100
                tau_ = tau(alpha(t), L{2}, opts.theta{2});
                [x{2}, p{2}] = ProxR{2}(tau_, xt - gk_u_kt/tau_, p{2});
                Fx{2} = F{2}(x{2});
                A_u_k = A(Fx{2}, Fx{1});

                data_u_k = objective.data(A_u_k);

                LHS1 = data_u_k;
                RHS1 = data_u_kt ...
                    + inner(gk_u_kt, x{2} - xt) ....
                    + 0.5 * L{2} * sqNorm(x{2} - xt) ...
                    + opts.ineq_tol;


               if LHS1 > RHS1 + opts.ineq_tol;
                    L{2} = min(L{2} * opts.eta_up{2}, opts.L_max{2});
                    continue;
                end

                Rk_k = objective.kernel(x{2});

                % test 3.7 (without convexity)
                % in notation of iPALM:
                % sigma(u+) <= sigma(u) + <u - u+, grad h(w)> + t/2 ||u - v||^2 - t/2 ||u+ - v||^2
                LHS2 = Rk_k;
                RHS2 = Rk_km + inner(xm{2} - x{2}, gk_u_kt) ...
                    + tau_/2 * (sqNorm(xm{2} - xt) - sqNorm(x{2} - xt)) ...
	            + opts.ineq_tol;	

                if LHS2 <= RHS2 + opts.ineq_tol;
                    L{2} = max(L{2} / opts.eta_down{2}, opts.L_min{2});
                    backtrackingSuccess = true;
                    break
                end
            end
            if ~backtrackingSuccess
                A_u_k = A_u_km;
                warning(['No step size found for kernel in iteration ', num2str(t)])
            end
            
        else
            A_u_k = A_u_km;
            x{2} = xm{2};
            Fx{2} = Fxm{2};
            data_u_k = data_u_km;
            Rk_k = Rk_km;
        end
         
        diffu = norm(Column(x{1} - xm{1})) / norm(Column(x{1}));
        diffk = norm(Column(x{2} - xm{2})) / norm(Column(x{2})); 

        xmm = xm;
        xm = x;
        
        Fxmm = Fxm;
        Fxm = Fx;

        A_um_km = A_u_k;       
        
        data_um_km = data_u_k;
        Ru_um = Ru_u;
        Rk_km = Rk_k;
                            
        % track progress                
        [objTrack, ukTrack] = RS_track_and_display_progress(objTrack, ...
            ukTrack, x{1}, x{2}, objective, opts.tracking_freq, ...
            opts.verbose_freq, opts.draw_iterates, opts.draw_iterates_fig, ...
            opts.niter, t, A_u_k, L{1}, L{2}, diffu, diffk);        
                
        if diffu < opts.tol
            fprintf('Iter %4d: Tolerance between image iterates reached. %2.2e < %2.2e\n', ...
                t , diffu, opts.tol);
            break;
        end
    end
end
