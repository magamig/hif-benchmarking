function [deltaA,deltaX] = tangent_space_L1_min_ipm(A,X)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% tangent_space_L1_min_ipm
%
%   Solves the optimization problem
%
%     min ||X + deltaX||_1   subj   A deltaX + deltaA X = 0,  
%                                   < A_i, deltaA_i >   = 0,  i = 1...n
%  
%   using an infeasible primal-dual interior point method that essentially
%   follows Mehrotra's algorithm (affine-scaling plus correction in each
%   step). 
%
%   Currently the method of choice in this toolbox.
%
%   Inputs:
%     A - m x n, represents the basis
%     X - n x p, represents the data coefficients wrt the basis 
%
%   Outputs:
%     deltaA -- step direction in A
%     deltaX -- step direction in X
%
%   Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    VERBOSE        = 3;
    ALPHA          = 0.05;
    BETA           = 0.5;
    MU             = 10;
    MAX_ITERATIONS = 15; 
    MAX_LINE_SEARCH_ITERATIONS = 50;

    [m,n] = size(A);
    [n,p] = size(X);

    mn = m*n; nm = mn; mp = m*p; np = n*p; 

    totalVars = 4*np+mn+mp+n; 

    % a few convenient handles for keeping things managable
    get_delta_x  = @(x) x(1:np);
    get_delta_a  = @(x) x(np+1:np+nm);
    get_w        = @(x) x(np+nm+1:2*np+nm);
    get_lambda_1 = @(x) x(2*np+nm+1:3*np+nm);
    get_lambda_2 = @(x) x(3*np+nm+1:4*np+nm);
    get_nu_1     = @(x) x(4*np+nm+1:4*np+nm+mp);
    get_nu_2     = @(x) x(4*np+nm+mp+1:4*np+nm+mp+n);

    put_delta_x  = @(x,y) [ y; x(np+1:end)];
    put_delta_a  = @(x,y) [ x(1:np); y; x(np+nm+1:end) ];
    put_w        = @(x,y) [ x(1:np+nm); y; x(2*np+nm+1:end) ];
    put_lambda_1 = @(x,y) [ x(1:2*np+nm); y; x(3*np+nm+1:end) ]; 
    put_lambda_2 = @(x,y) [ x(1:3*np+nm); y; x(4*np+nm+1:end) ];
    put_nu_1     = @(x,y) [ x(1:4*np+nm); y; x(4*np+nm+mp+1:end) ];
    put_nu_2     = @(x,y) [ x(1:4*np+nm+mp); y ];

    eval_f1 = @(z)  vec(X) + get_delta_x(z) - get_w(z);
    eval_f2 = @(z) -vec(X) - get_delta_x(z) - get_w(z);

    eval_L1 = @(z) vec( A * reshape(get_delta_x(z),n,p) + reshape(get_delta_a(z),m,n) * X );
    eval_L2 = @(z) CA_transpose( A, reshape(get_delta_a(z),m,n) );

    eval_L_adjoint = @(nu1,nu2) [ vec(A'* reshape(nu1,m,p)); vec(reshape(nu1,m,p)*X' + CA(A,nu2)); zeros(np,1) ];

    eval_r_dual    = @(z) [ zeros(np+mn,1); ones(np,1) ] + [ get_lambda_1(z) - get_lambda_2(z); zeros(mn,1); -get_lambda_1(z) - get_lambda_2(z) ] + eval_L_adjoint(get_nu_1(z),get_nu_2(z));
    eval_r_central = @(z) [ -get_lambda_1(z) .* eval_f1(z); -get_lambda_2(z) .* eval_f2(z) ];
    eval_r_primal  = @(z)[ eval_L1(z) ; eval_L2(z) ];

    eval_sdg = @(z) (-eval_f1(z)' * get_lambda_1(z) - eval_f2(z)' * get_lambda_2(z)) / (2*n*p);
    eval_rc_correction = @(delta,mu,sigma) -sigma*mu*ones(2*np,1) - [ get_delta_x(delta) - get_w(delta); -get_delta_x(delta) - get_w(delta) ] .* [ get_lambda_1(delta); get_lambda_2(delta) ];    

    % initialization: 
    x_pd = zeros(totalVars,1);

    x_pd = put_w(x_pd, vec(1.5*abs(X)));
    x_pd = put_lambda_1(x_pd, ones(np,1));
    x_pd = put_lambda_2(x_pd, ones(np,1));

    eta_hat = eval_sdg(x_pd);

    numIterations = 0;
    converged = false;

    while ~converged

        numIterations = numIterations + 1;
        
        mu = eval_sdg(x_pd);

        % set up the right hand side (calcluate residuals)
        r_dual     =  eval_r_dual(x_pd); 
        r_central  =  eval_r_central(x_pd);
        r_primal   =  eval_r_primal(x_pd);

        delta_affine_scaling = solve_step_equations_new_reduction;
        delta_affine_scaling = solve_step_equations_2x2_preconditioned;
        
        % calcluate the max primal and dual step size for feasibility of
        %  the inequality constraints
        alpha_primal_as   = compute_primal_step(x_pd,delta_affine_scaling);
        alpha_dual_as     = compute_dual_step(x_pd,delta_affine_scaling);
        
        % duality gap after PD step
        x_pd_as = apply_pd_step(x_pd,delta_affine_scaling,alpha_primal_as,alpha_dual_as);        
        mu_aff  = eval_sdg( x_pd_as );        
        sigma   = (mu_aff / mu)^3;
        
        % corrector step
        r_dual = zeros(2*np+mn,1);
        r_primal = zeros(mp+n,1);
        r_central = eval_rc_correction(delta_affine_scaling,mu,sigma);
        
        delta_corrector = solve_step_equations_2x2_preconditioned;
        
        delta = delta_affine_scaling + delta_corrector;
        
        alpha_primal_max = compute_primal_step(x_pd,delta);
        alpha_dual_max   = compute_dual_step(x_pd,delta);
        
        alpha_primal = min( .99 * alpha_primal_max, 1 );
        alpha_dual   = min( .99 * alpha_dual_max,   1 );
                      
        x_pd = apply_pd_step(x_pd,delta,alpha_primal,alpha_dual);

        % extraneous, just for easy output... 
        r_dual_new    = eval_r_dual(x_pd);
        r_central_new = eval_r_central(x_pd);
        r_primal_new  = eval_r_primal(x_pd);        

        newResidualNorm = sqrt( r_dual_new'*r_dual_new + r_central_new'*r_central_new + r_primal_new'*r_primal_new );
        
        % check convergence: duality gap or number of iterations
        eta_hat = eval_sdg(x_pd); 

        if VERBOSE > 1, 
            disp(['        Interior point iteration ' num2str(numIterations) '   SDG: ' num2str(eta_hat) '   PD Res: ' num2str(newResidualNorm)]);
        end 
        
        disp(['          Checking feasibility ...  min lambda: ' num2str(min([get_lambda_1(x_pd); get_lambda_2(x_pd)])) '   max f: ' num2str(max([eval_f1(x_pd); eval_f2(x_pd)]))]);

        converged = numIterations >= MAX_ITERATIONS;
        
        %pause;
        
    end     

    deltaA = reshape(get_delta_a(x_pd),m,n);
    deltaX = reshape(get_delta_x(x_pd),n,p);
    
    if VERBOSE > 0,
        disp(['        Interior point terminated with SDG: ' num2str(eta_hat) '   PD Residual: ' num2str(newResidualNorm)]);
    end
    
    return;
    
    function alpha_pr = compute_primal_step(x,delta)
        
        f1 = eval_f1(x);
        f2 = eval_f2(x);
        f = [ f1; f2 ];        
        
        v1 =     get_delta_x(delta) - get_w(delta);
        v2 =   - get_delta_x(delta) - get_w(delta);
        v = [ v1; v2 ];        
        J = find(v > 0);       
        
        alpha_pr = min( -f(J) ./ v(J) );
        
    end

    function alpha_dual = compute_dual_step(x,delta)
        
        lambda = [ get_lambda_1(x); get_lambda_2(x) ];
        delta_lambda = [ get_lambda_1(delta); get_lambda_2(delta) ];        
        J = find(delta_lambda < 0);
        
        alpha_dual = min( -lambda(J) ./ delta_lambda(J) );
        
    end

    function x_new = apply_pd_step(x,delta,alphaPrimal,alphaDual)
        
        x_new = zeros(totalVars,1);
        x_new = put_delta_x(x_new, get_delta_x(x) + alphaPrimal * get_delta_x(delta));
        x_new = put_delta_a(x_new, get_delta_a(x) + alphaPrimal * get_delta_a(delta));
        x_new = put_w(x_new, get_w(x) + alphaPrimal * get_w(delta));
        x_new = put_lambda_1(x_new, get_lambda_1(x) + alphaDual * get_lambda_1(delta));
        x_new = put_lambda_2(x_new, get_lambda_2(x) + alphaDual * get_lambda_2(delta));
        x_new = put_nu_1(x_new, get_nu_1(x) + alphaDual * get_nu_1(delta));
        x_new = put_nu_2(x_new, get_nu_2(x) + alphaDual * get_nu_2(delta));
        
    end

    function delta_x_pd = solve_step_equations_new_reduction
        
        % dense solution after reduction to four vector unknowns, 
        %  delta_DX, delta_DA, delta_nu1, delta_nu2        
        C_A = zeros(mn,n); 
        for i = 1:n,
            C_A(m*(i-1)+1:m*i,i) = A(:,i);
        end

        % precomputations
        lambda1 = get_lambda_1(x_pd);
        lambda2 = get_lambda_2(x_pd);    
        f1 = eval_f1(x_pd);
        f2 = eval_f2(x_pd);    
        nu1 = get_nu_1(x_pd);
        nu2 = get_nu_2(x_pd);            
        
        % compute gamma and gamma inverse
        gamma11 = -lambda1;
        gamma12 = .5 * ( lambda1 - f1 );
        gamma21 =  lambda2;
        gamma22 = .5 * ( lambda2 + f2 );
        
        det_gamma = gamma11 .* gamma22 - gamma21 .* gamma12;
        gammaInv11 =  gamma22 ./ det_gamma;
        gammaInv12 = -gamma12 ./ det_gamma;
        gammaInv21 = -gamma21 ./ det_gamma;
        gammaInv22 =  gamma11 ./ det_gamma;
        
        u1  = .5 * (lambda1 + f1);
        u2  = .5 * (lambda2 - f2);
        
        phi = gammaInv21 .* u1 + gammaInv22 .* u2; 
        psi = gammaInv11 .* u1 + gammaInv12 .* u2; 
        
        disp(['Phi min and max: ' num2str(min(abs(phi))) ' ' num2str(max(abs(phi)))]);
                
        delta_x_pd = zeros(totalVars,1);
        
    end

    function delta_x_pd = solve_step_equations_2x2
        
        U = zeros(mn+n,mn+n);
        
        % dense solution after reduction to four vector unknowns, 
        %  delta_DX, delta_DA, delta_nu1, delta_nu2        
        C_A = zeros(mn,n); 
        for i = 1:n,
            C_A(m*(i-1)+1:m*i,i) = A(:,i);
        end

        % precomputations
        lambda1 = get_lambda_1(x_pd);
        lambda2 = get_lambda_2(x_pd);    
        f1 = eval_f1(x_pd);
        f2 = eval_f2(x_pd);    
        nu1 = get_nu_1(x_pd);
        nu2 = get_nu_2(x_pd);            
                
        % invert gamma
        g_det = lambda1 .* f2 + lambda2 .* f1;        
        g_inv_11 =       f2 ./ g_det;   g_inv_12 =      f1 ./ g_det;
        g_inv_21 = -lambda2 ./ g_det;   g_inv_22 = lambda1 ./ g_det;             
        
        phi = g_inv_21 .* lambda1 - g_inv_22 .* lambda2;
        
        gamma_inv_norm_max = 0;
        for ii = 1:p,
            Gamma_inv_cur = zeros(2,2);
            Gamma_inv_cur(1,1) = g_inv_11(ii);
            Gamma_inv_cur(1,2) = g_inv_12(ii);
            Gamma_inv_cur(2,1) = g_inv_21(ii);
            Gamma_inv_cur(2,2) = g_inv_22(ii);
            
            gamma_s = svd(Gamma_inv_cur);
            
            if gamma_s(1) > gamma_inv_norm_max,
                gamma_inv_norm_max = gamma_s(1);
            end
        end
        disp(['Maximum norm of Gamma inverse block: ' num2str(gamma_inv_norm_max)]);
        disp(['Min of first hypothetical preconditioner: ' num2str(min(lambda1 - f1))]);
        disp(['Min of second hypothetical preconditioner: ' num2str(min(lambda2 - f2))]);
        
        phi_max = max(abs(phi));
        phi_min = min(abs(phi));
        disp(['Conditioning of Phi ... max: ' num2str(phi_max) '   min: ' num2str(phi_min) '   ratio: ' num2str(phi_max/phi_min)]);
        
        Psi_blocks = cell(p,1);
        Psi_inv_blocks = cell(p,1);
        for ii = 1:p,
            Psi_blocks{ii} = - .5 * A * inv(diag(phi(n*(ii-1)+1:n*ii))) * A';
            Psi_inv_blocks{ii} = inv(Psi_blocks{ii});
        end        
        
        % fill in U
        % V = - kron(X,eye(m)) * Psi_inv * kron(X',eye(m));
        V = zeros(mn,mn);
        for ii = 1:n,
            for jj = ii:n,
                for kk = 1:p,
                    V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj) = V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj) - (X(ii,kk) * X(jj,kk)) * Psi_inv_blocks{kk};
                end
                V(m*(jj-1)+1:m*jj,m*(ii-1)+1:m*ii) = V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj);
            end
        end
        
        U(1:mn,1:mn) = V; 
        U(1:mn,mn+1:mn+n) = C_A;
        U(mn+1:mn+n,1:mn) = C_A'; 
        
        % RHS               
        r_d1 = r_dual(1:np);
        r_d2 = r_dual(np+1:np+nm);
        r_d3 = r_dual(np+nm+1:2*np+nm);
        
        r_c1 = r_central(1:np);
        r_c2 = r_central(np+1:2*np);
        
        r_p1 = r_primal(1:mp);
        r_p2 = r_primal(mp+1:mp+n);
        
        u1 = -g_inv_11 .* r_c1 + g_inv_12 .* ( f2 .* r_d3 - r_c2 );
        u2 = -g_inv_21 .* r_c1 + g_inv_22 .* ( f2 .* r_d3 - r_c2 );        
        
        v1 =  r_d3 - r_d1 - 2 * u2;
        v2 = -r_d2;
        v3 = -r_p1;
        v4 = -r_p2;
        
        y1 = v2;
        y2 = v3 - .5 * vec( A * reshape(v1 ./ phi,n,p) );
        y3 = v4; 
        
        Psi_inv_y2 = zeros(mp,1);
        for ii = 1:p,
            Psi_inv_y2(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * y2(m*(ii-1)+1:m*ii); 
        end
        z1 = y1 - vec( reshape(Psi_inv_y2,m,p) * X' );
        z2 = y3; 
        
        z = [ z1; z2 ];
        
        delta = inv(U) * z; 
        
        delta_dA  = delta(1:mn);
        delta_nu2 = delta(mn+1:mn+n);                
        delta_nu1 = zeros(mp,1);        
        qq = y2 - vec(reshape(delta_dA,m,n) * X);
        for ii = 1:p,
            delta_nu1(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * qq(m*(ii-1)+1:m*ii);
        end        
        delta_dX      = .5 * ( v1 - vec(A' * reshape(delta_nu1,m,p)) ) ./ phi;         
        delta_W       = ( g_inv_11 .* lambda1 - g_inv_12 .* lambda2 ) .* delta_dX + u1;
        delta_lambda1 = ( g_inv_21 .* lambda1 - g_inv_22 .* lambda2 ) .* delta_dX + u2;        
        delta_lambda2 =   r_d3 - delta_lambda1;        
        
        delta_x_pd = [ delta_dX; delta_dA; delta_W; delta_lambda1; delta_lambda2; delta_nu1; delta_nu2 ];        
    end

    function delta_x_pd = solve_step_equations_2x2_preconditioned
        
        U = zeros(mn+n,mn+n);
        
        % dense solution after reduction to four vector unknowns, 
        %  delta_DX, delta_DA, delta_nu1, delta_nu2        
        C_A = zeros(mn,n); 
        for i = 1:n,
            C_A(m*(i-1)+1:m*i,i) = A(:,i);
        end

        % precomputations
        lambda1 = get_lambda_1(x_pd);
        lambda2 = get_lambda_2(x_pd);    
        f1 = eval_f1(x_pd);
        f2 = eval_f2(x_pd);    
        nu1 = get_nu_1(x_pd);
        nu2 = get_nu_2(x_pd);            
        
        pc1 = lambda1 - f1;
        pc2 = lambda2 - f2;
        
        lambda1 = lambda1 ./ pc1;
        lambda2 = lambda2 ./ pc2;
        f1 = f1 ./ pc1;
        f2 = f2 ./ pc2;
                
        % invert gamma
        g_det = lambda1 .* f2 + lambda2 .* f1;        
        g_inv_11 =       f2 ./ g_det;   g_inv_12 =      f1 ./ g_det;
        g_inv_21 = -lambda2 ./ g_det;   g_inv_22 = lambda1 ./ g_det;             
        
        phi = g_inv_21 .* lambda1 - g_inv_22 .* lambda2;          
        
        gamma_inv_norm_max = 0;
        gamma_new_kappa_max = 1;
        
        for ii = 1:p,
            Gamma_inv_cur = zeros(2,2);
            Gamma_inv_cur(1,1) = g_inv_11(ii);
            Gamma_inv_cur(1,2) = g_inv_12(ii);
            Gamma_inv_cur(2,1) = g_inv_21(ii);
            Gamma_inv_cur(2,2) = g_inv_22(ii);
            
            gamma_s = svd(Gamma_inv_cur);
            
            Gamma_new_cur = zeros(2,2);
            Gamma_new_cur(1,1) = -lambda1(ii);  Gamma_new_cur(1,2) = .5 * ( lambda1(ii) - f1(ii) );
            Gamma_new_cur(2,1) =  lambda2(ii);  Gamma_new_cur(2,2) = .5 * ( lambda2(ii) + f2(ii) ); 
            
            gamma_new_s = svd(Gamma_new_cur);
            kappa_new = gamma_new_s(1) / gamma_new_s(end);
            
            if gamma_s(1) > gamma_inv_norm_max,
                gamma_inv_norm_max = gamma_s(1);
            end
            
            if kappa_new > gamma_new_kappa_max,
                gamma_new_kappa_max = kappa_new;
            end
        end
        disp(['Maximum norm of Gamma inverse block: ' num2str(gamma_inv_norm_max)]);
        disp(['Max condition number for new Gamma: ' num2str(gamma_new_kappa_max)]);
        disp(['Min of first hypothetical preconditioner: ' num2str(min(lambda1 - f1))]);
        disp(['Min of second hypothetical preconditioner: ' num2str(min(lambda2 - f2))]);
        
        phi_max = max(abs(phi));
        phi_min = min(abs(phi));
        disp(['Conditioning of Phi ... max: ' num2str(phi_max) '   min: ' num2str(phi_min) '   ratio: ' num2str(phi_max/phi_min)]);
        
        Psi_blocks = cell(p,1);
        Psi_inv_blocks = cell(p,1);
        for ii = 1:p,
            Psi_blocks{ii} = - .5 * A * inv(diag(phi(n*(ii-1)+1:n*ii))) * A';
            Psi_inv_blocks{ii} = inv(Psi_blocks{ii});
        end        
        
        % fill in U
        % V = - kron(X,eye(m)) * Psi_inv * kron(X',eye(m));
        V = zeros(mn,mn);
        for ii = 1:n,
            for jj = ii:n,
                for kk = 1:p,
                    V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj) = V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj) - (X(ii,kk) * X(jj,kk)) * Psi_inv_blocks{kk};
                end
                V(m*(jj-1)+1:m*jj,m*(ii-1)+1:m*ii) = V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj);
            end
        end
        
        U(1:mn,1:mn) = V; 
        U(1:mn,mn+1:mn+n) = C_A;
        U(mn+1:mn+n,1:mn) = C_A'; 
        
        % RHS               
        r_d1 = r_dual(1:np);
        r_d2 = r_dual(np+1:np+nm);
        r_d3 = r_dual(np+nm+1:2*np+nm);
        
        r_c1 = r_central(1:np);
        r_c2 = r_central(np+1:2*np);
        
        r_c1 = r_c1 ./ pc1;
        r_c2 = r_c2 ./ pc2;
        
        r_p1 = r_primal(1:mp);
        r_p2 = r_primal(mp+1:mp+n);
        
        u1 = -g_inv_11 .* r_c1 + g_inv_12 .* ( f2 .* r_d3 - r_c2 );
        u2 = -g_inv_21 .* r_c1 + g_inv_22 .* ( f2 .* r_d3 - r_c2 );        
        
        v1 =  r_d3 - r_d1 - 2 * u2;
        v2 = -r_d2;
        v3 = -r_p1;
        v4 = -r_p2;
        
        y1 = v2;
        y2 = v3 - .5 * vec( A * reshape(v1 ./ phi,n,p) );
        y3 = v4; 
        
        Psi_inv_y2 = zeros(mp,1);
        for ii = 1:p,
            Psi_inv_y2(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * y2(m*(ii-1)+1:m*ii); 
        end
        z1 = y1 - vec( reshape(Psi_inv_y2,m,p) * X' );
        z2 = y3; 
        
        z = [ z1; z2 ];
        
        delta = inv(U) * z; 
        
        delta_dA  = delta(1:mn);
        delta_nu2 = delta(mn+1:mn+n);                
        delta_nu1 = zeros(mp,1);        
        qq = y2 - vec(reshape(delta_dA,m,n) * X);
        for ii = 1:p,
            delta_nu1(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * qq(m*(ii-1)+1:m*ii);
        end        
        delta_dX      = .5 * ( v1 - vec(A' * reshape(delta_nu1,m,p)) ) ./ phi;         
        delta_W       = ( g_inv_11 .* lambda1 - g_inv_12 .* lambda2 ) .* delta_dX + u1;
        delta_lambda1 = ( g_inv_21 .* lambda1 - g_inv_22 .* lambda2 ) .* delta_dX + u2;        
        delta_lambda2 =   r_d3 - delta_lambda1;        
        
        delta_x_pd = [ delta_dX; delta_dA; delta_W; delta_lambda1; delta_lambda2; delta_nu1; delta_nu2 ];        
    end
end

