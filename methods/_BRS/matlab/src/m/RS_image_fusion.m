function [u_star, u_inner, k_star, iter, data, Ax, objTrack, ukTrack] = ...
    RS_image_fusion(dataset, param_model, param_alg)
%RS_image_fusion Solves problem (1) from our paper.
% 
% Input: 
%     dataset [string]
%         name of the data set
%     param_model [struct]
%         model parameters
%     param_alg [struct]
%         algorithm parameters 
%         
% Output:
%     u_star [matrix]
%         optimal image
%     u_inner [matrix]
%         meaningful part of the optimal image
%     k_star [matrix]
%         optimal blurring kernel
%     iter [int]
%         number of PALM iterations, 
%     data [matrix]
%         hyperspectral data image
%     Ax [matrix]
%         application of the forward operator to optimal image and kernel
%     objTrack [matrix]
%         information about objective function values, number of ffts, time, etc.
%     ukTrack [cell]
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

    downsampling_factor = 4; % downsampling factor

    % kernel and margin
    margin = 20;
       
    if ~isfield(param_alg, 'tracking_freq')
        param_alg.tracking_freq.trackUk = 0;
        param_alg.tracking_freq.trackObj = 0;        
    end
    
    %% kernel parameters
    s_kernel  = (2*margin + 1) * [1, 1];

    % set up data
    load(dataset);
    
    if size(side_info, 3) > 1
        side_info = rgb2gray(side_info);
    end
   
    s_image = size(side_info) + 2*margin;
    
    % boundary treatment
    side_info = padarray(side_info, [margin, margin], 'symmetric');
            
    %% define subsampling and blowup indices
    blowupInd = RS_embedding_indices(s_kernel, s_image);
    
    % define involved operators and their adjoints
    % embedding operator
    J = @(x) RS_pad_array(x, s_image);
    Jadj = @(x) reshape(x(blowupInd), s_kernel);
    % downsampling operator
    S = @(x) RS_downsampling(x, downsampling_factor,margin); 
    Sadj = @(y) RS_downsampling_adjoint(y, downsampling_factor,margin);
    SadjSInv = @(x,rho) RS_downsampling_adjoint_invers(x, downsampling_factor, rho, S, Sadj);
    % convolution operator accepting data in Fourier domain
    C = @(Fk, Fu) real(RS_fourier_inverse(Fk .* Fu));
    Cadj = @(Fk, Fu) real(RS_fourier_inverse(conj(Fk).*Fu));
    % combined forward operator
    A = @(Fk, Fu) S(C(Fk, Fu));
    % combined Fourier transformation
    F = {@(x) RS_fourier(x), @(x) RS_fourier(J(x))};
   
    % define proximal operators
    vfield = RS_vectorfield(side_info, param_model.eps, param_model.gamma);
       
    opts_ProxRu.PC = @(x) max(x,0);
    opts_ProxRk.PC = @(k) reshape(RS_ESP(k),s_kernel);
    opts_ProxRu.tol = 1e-3;
    opts_ProxRk.tol = 1e-3;
    opts_ProxRu.min_iter = 20;
    opts_ProxRk.min_iter = 20;

    switch param_alg.algorithm
        case {'PALM'}
            opts_ProxRu.max_iter = 20; %15?
            opts_ProxRk.max_iter = 20; %15?
        case {'PAM'}
            opts_ProxRu.max_iter = 200; %15?
            opts_ProxRk.max_iter = 200; %15?
    end
    
    ProxRu = @(t, u, p) RS_FGP_dTV(u, param_model.lambda_u/t, opts_ProxRu.max_iter, ...
            vfield, opts_ProxRu, p);
    
    if param_model.lambda_k > 0
        ProxRk = @(t, k, q) RS_FGP_TV(k, param_model.lambda_k/t, opts_ProxRk.max_iter, ...
            opts_ProxRk, q); %TV plus simplex projection
    else
        ProxRk = @(t, k, q) deal(reshape(RS_ESP(k),s_kernel),q);   % no regularization, just simplex projection
    end 
    ProxR = {ProxRu, ProxRk};

    % Gradient of D
    gradD = {@(Fk, Auk) Cadj(Fk,RS_fourier(Sadj(Auk - data))), ...
             @(Fu, Auk) Jadj(Cadj(Fu,RS_fourier(Sadj(Auk - data))))};

    % Lipschitz constants
    L = {1, 1};  

    % initialize u and k
    u_init = RS_downsampling_right_invers(data, downsampling_factor, margin); %S(u_init) = data; 
    k_init = RS_initialize_kernel(s_kernel);
    k_init = k_init / sum(k_init(:)); % redundant
    x_init = {u_init, k_init};

    %% add function handle for evaluation of minimization problem
    objective.image = @(u) RS_dTV(u, vfield, param_model.lambda_u); 
    objective.kernel = @(k) RS_TV(k, param_model.lambda_k); 
    objective.data = @(A_u_k) RS_data_fidelity(A_u_k, data); 

    %% Solve minimization problem with palm
    if param_alg.draw_iterates
        opts.draw_iterates_fig = figure();
    else
        opts.draw_iterates_fig = [];
    end
    
    switch param_alg.algorithm
        case {'PALM'}
        
            % small constant greater 1 for step sizes
            opts.theta = {1.1, 1.1};
            opts.L_min = {1e-0, 1e-0};
            opts.L_max = {1e+30, 1e+30};

            % updating parameters for backtracking
            opts.eta_down = {2, 2};
            opts.eta_up = {2, 2};
            
            % inertial parameter
            opts.inertia_alpha = param_alg.inertia;            
            if isnumeric(param_alg.inertia) && param_alg.inertia == 0
                opts.inertia = false;
            else
                opts.inertia = true;
            end
            
            % misc
            opts.tol = 1e-5;
            opts.ineq_tol = 1e-6;
            
        case {'PAM'}
            opts.tol = 1e-8;
            opts.maxIter = param_alg.niter;
            opts.tolProxU = 1e-3;
            opts.tolProxK = 1e-3;
            opts.minIterProxU = 5;
            opts.minIterProxK = 5;     
            opts.maxIterProxU = 200;
            opts.maxIterProxK = 200;     
            opts.useInertial = false;    
            opts.tu = 1; 
            opts.tk = 1;     
            opts.rho_k = 1.2;
            operators.A = A;
            operators.F = F;
            operators.C = C;
            operators.J = J;
            operators.Jadj = Jadj;
            operators.SadjSInv = SadjSInv;
    end
    
    opts.draw_iterates = param_alg.draw_iterates;
    opts.tracking_freq = param_alg.tracking_freq;
    opts.niter = param_alg.niter;
    opts.verbose_freq = param_alg.verbose_freq;
    opts.update_kernel = param_alg.blind;
    
    tic;
    switch param_alg.algorithm
        case 'PALM'
            [x_star, Ax, iter, objTrack, ukTrack] = RS_PALM( ...
                x_init, A, F, gradD, L, ProxR, objective, opts); 
        case 'PAM'
            [x_star, Ax, iter, objTrack, ukTrack] = RS_PAM( ...
                x_init, Sadj(data), operators, ProxR,...
                objective, opts);    
    end
    
    u_star = x_star{1};
    k_star = x_star{2};
    
    % crop artificial boundary
    u_inner = u_star((1+margin):(end-margin),(1+margin):(end-margin)); 
    
    % show iterates
    if opts.draw_iterates
        close(opts.draw_iterates_fig);
    
        figure();
        subplot(1,2,1);
        imagesc(u_star);
        colorbar; colormap gray; axis tight; axis equal; title('u*');
        subplot(1,2,2);
        imagesc(k_star);
        colorbar; colormap gray; axis tight; axis equal; title('k*');
    end    
 end
