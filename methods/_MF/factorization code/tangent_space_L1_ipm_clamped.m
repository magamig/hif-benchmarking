function [deltaA,deltaX] = tangent_space_L1_ipm_clamped(A,X,eps)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% tangent_space_L1_min_interior_point
%
%   Solves the optimization problem
%
%     min ||X + deltaX||_1   subj   A deltaX + deltaA X = 0,  
%                                   < A_i, deltaA_i >   = 0,  i = 1...n
%                                   ||deltaA||_F^2 + ||deltaX||_F^2 <= eps^2
%  
%     using Mehrotra's centering approach, 
%     with a few tricks to solve the Newton system more efficiently. 
%
%   Inputs:
%     A - m x n, presumably an estimate of a sparsifying basis
%     X - n x p, presumably an estimate of sparse coefficients wrt the basis
%     eps - scalar clamp parameter; maximum allowable step length
%
%   Outputs:
%     deltaA -- step direction in A
%     deltaX -- step direction in X
%
%   Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%  %%%%%%%%%%%%%%% INCOMPLETE! %%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

	error('tangent_space_L1_ipm_clamped: this function is not yet complete!');

	VERBOSE        = 3;
    MAX_ITERATIONS = 50; 

    [m,n] = size(A);
    [n,p] = size(X);

    mn = m*n; nm = mn; mp = m*p; np = n*p; 

    totalVars = 4*np+mn+mp+n+1; 

    % a few convenient handles for keeping things managable
    get_delta_x  = @(x) x(1:np);
    get_delta_a  = @(x) x(np+1:np+nm);
    get_w        = @(x) x(np+nm+1:2*np+nm);
    get_lambda_1 = @(x) x(2*np+nm+1:3*np+nm);
    get_lambda_2 = @(x) x(3*np+nm+1:4*np+nm);
	get_lambda_3 = @(x) x(4*np+nm+1);
    get_nu_1     = @(x) x(4*np+nm+2:4*np+nm+mp+1);
    get_nu_2     = @(x) x(4*np+nm+mp+2:4*np+nm+mp+n+1);

    eval_f1 = @(z)  vec(X) + get_delta_x(z) - get_w(z);
    eval_f2 = @(z) -vec(X) - get_delta_x(z) - get_w(z);
	eval_f3 = @(z) norm(get_delta_a(z))^2 + norm(get_delta_x(z))^2 - eps^2;
    eval_L1 = @(z) vec( A * reshape(get_delta_x(z),n,p) + reshape(get_delta_a(z),m,n) * X );
    eval_L2 = @(z) CA_transpose( A, reshape(get_delta_a(z),m,n) );
    eval_L_adjoint = @(nu1,nu2) [ vec(A'* reshape(nu1,m,p)); vec(reshape(nu1,m,p)*X' + CA(A,nu2)); zeros(np,1) ];
    eval_r_dual    = @(z) [ zeros(np+mn,1); ones(np,1) ] + [ get_lambda_1(z) - get_lambda_2(z); zeros(mn,1); -get_lambda_1(z) - get_lambda_2(z) ] + eval_L_adjoint(get_nu_1(z),get_nu_2(z)) + 2 * get_lambda_3(z) * [ get_delta_x(z); get_delta_a(z); zeros(np,1) ];
    eval_r_central = @(z) [ -get_lambda_1(z) .* eval_f1(z); -get_lambda_2(z) .* eval_f2(z); -get_lambda_3(z) * eval_f3(z) ];
    eval_r_primal  = @(z)[ eval_L1(z) ; eval_L2(z) ];
    eval_sdg  = @(z) (- eval_f1(z)' * get_lambda_1(z) - eval_f2(z)' * get_lambda_2(z) - eval_f3(z) * get_lambda_3(z) ) / (2*n*p+1);
    eval_rc_correction = @(delta,mu,sigma) -sigma*mu*ones(2*np+1,1) - [ get_delta_x(delta) - get_w(delta); -get_delta_x(delta) - get_w(delta); 0 ] .* [ get_lambda_1(delta); get_lambda_2(delta); get_lambda_3(delta) ];    
	

    % initialization: 
    x_pd = zeros(totalVars,1);
	x_pd(np+nm+1:2*np+nm) = vec(1.5*abs(X));
	x_pd(2*np+nm+1:4*np+nm+1) = ones(2*np+1,1);

    eta_hat = eval_sdg(x_pd);

    numIterations = 0;
    converged = false;

    while ~converged
	
        numIterations = numIterations + 1;

		mu = eval_sdg(x_pd);
		
		delta_x = get_delta_x(x_pd); delta_a = get_delta_a(x_pd); lambda1 = get_lambda_1(x_pd); lambda2 = get_lambda_2(x_pd); lambda3 = get_lambda_3(x_pd);	f1 = eval_f1(x_pd);	f2 = eval_f2(x_pd);	f3 = eval_f3(x_pd);	nu1 = get_nu_1(x_pd); nu2 = get_nu_2(x_pd);		
		
		% affine scaling step - pure Newton step for the KKT equations		
        r_dual     =  eval_r_dual(x_pd); 
        r_central  =  eval_r_central(x_pd);
        r_primal   =  eval_r_primal(x_pd);

		% solve affine scaling step		
        delta_x_affine_scaling = solve_step_equations(A,X,delta_x,delta_a,lambda1,lambda2,lambda3,f1,f2,f3,nu1,nu2,r_primal,r_central,r_dual);

		% max step sizes that maintain primal and dual feasibility
        alpha_primal_as   = compute_primal_step(delta_x,delta_a,w,vec(X),get_delta_x(delta_x_affine_scaling),get_delta_a(delta_x_affine_scaling),get_w(delta_x_affine_scaling),eps);
        alpha_dual_as     = compute_dual_step(lambda1,lambda2,lambda3,get_lambda_1(delta_x_affine_scaling),get_lambda_2(delta_x_affine_scaling),get_lambda_3(delta_x_afine_scaling));
		
        % duality gap after PD step
        x_pd_as = zeros(totalVars,1);
		x_pd_as(1:2*np+mn)     = x_pd(1:2*np+mn)     + alpha_primal_as * delta_affine_scaling(1:2*np+mn);
		x_pd_as(2*np+mn+1:end) = x_pd(2*np+mn+1:end) + alpha_dual_as   * delta_affine_scaling(2*np+mn+1:end);
		
        mu_aff  = eval_sdg( x_pd_as );        
        sigma   = (mu_aff / mu)^3;		
		
        % corrector step
        r_dual    = zeros(2*np+mn,1);
        r_primal  = zeros(mp+n,1);
        r_central = eval_rc_correction(delta_affine_scaling,mu,sigma);
        
        delta_corrector = solve_step_equations(A,X,get_delta_x(x_pd_as),get_delta_a(x_pd_as),get_lambda_1(x_pd_as),get_lambda_2(x_pd_as),eval_f1(x_pd_as),eval_f2(x_pd_as),eval_f3(x_pd_as),get_nu_1(x_pd_as),get_nu_2(x_pd_as),r_primal,r_central,r_dual);    
        delta = delta_affine_scaling + delta_corrector;
        
        alpha_primal_max = compute_primal_step(delta_x,delta_a,w,vec(X),get_delta_x(delta),get_delta_a(delta),get_w(delta),eps);
        alpha_dual_max   = compute_dual_step(lambda1,lambda2,lambda3,get_lambda_1(delta),get_lambda_2(delta),get_lambda_3(delta));
        
        alpha_primal = min( .99 * alpha_primal_max, 1 );
        alpha_dual   = min( .99 * alpha_dual_max,   1 );
                      
        x_pd(1:2*np+mn)     = x_pd(1:2*np+mn)     + alpha_primal * delta(1:2*np+mn);
		x_pd(2*np+mn+1:end) = x_pd(2*np+mn+1:end) + alpha_dual   * delta(2*np+mn+1:end);
		
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
      
        % check convergence: duality gap or number of iterations
        eta_hat = eval_sdg(x_pd); 

        if VERBOSE > 1, 
            disp(['        Interior point iteration ' num2str(numIterations) '   SDG: ' num2str(eta_hat / (2*np+1)) '   PD Res: ' num2str(newResidualNorm) '   t: ' num2str(t) ]);
        end 

        converged = numIterations >= MAX_ITERATIONS;

    end     

    deltaA = reshape(get_delta_a(x_pd),m,n);
    deltaX = reshape(get_delta_x(x_pd),n,p);
    
    if VERBOSE > 0,
        disp(['        Interior point terminated with SDG: ' num2str(eta_hat) '   PD Residual: ' num2str(newResidualNorm)]);
    end
    
    return;

end

function alpha_primal = compute_primal_step(delta_x,delta_a,w,x,delta_dx,delta_da,delta_w,eps)

	f1 =  x + delta_x - w;
	f2 = -x - delta_x - w;
	f = [ f1; f2 ];        
	
	v1 =  delta_dx - delta_w;
	v2 = -delta_dx - delta_w;
	v = [ v1; v2 ];        
	J = find(v > 0);       
	
	alpha_pr = min( -f(J) ./ v(J) );

	a = norm( delta_dx )^2 + norm(delta_da)^2;
	b = 2 * ( delta_dx' * delta_x + delta_da' * delta_a );
	c = norm(delta_x)^2 + norm(delta_a)^2 - eps^2;	
	alphaBall = (- b + sqrt(b^2-4*a*c)) / (2 * a);	
	
	alpha_primal = min(alpha_primal,alphaBall);
end

function alpha_dual = compute_dual_step(lambda1,lambda2,lambda3,delta_lambda1,delta_lambda2,delta_lambda3)

	lambda = [ lambda1; lambda2; lambda3 ];
	delta_lambda = [ delta_lambda1; delta_lambda2; delta_lambda3 ];        
	J = find(delta_lambda < 0);
	alpha_dual = min( -lambda(J) ./ delta_lambda(J) );
	
end


function delta_x_pd = solve_step_equations(A,X,delta_x,delta_a,lambda1,lambda2,lambda3,f1,f2,f3,nu1,nu2,r_primal,r_central,r_dual)

	[m,n] = size(A);
	[n,p] = size(X);
	
	mn = m*n; nm = n*m; mp = m*p; np = n*p;
	
	totalVars = 4*np+mn+mp+n+1;

	% dense solution after reduction to four vector unknowns, 
	%  delta_DX, delta_DA, delta_nu1, delta_nu2        
	C_A = zeros(mn,n); 
	for i = 1:n,
		C_A(m*(i-1)+1:m*i,i) = A(:,i);
	end

	
	%%%
	%   Solve using complete reduction to system of size mn+n+1,
	%      plus additional optimizations for scalability
	%%%
	
	% invert gamma
	g_det    = lambda1 .* f2 + lambda2 .* f1;        
	g_inv_11 =       f2 ./ g_det;   g_inv_12 =      f1 ./ g_det;
	g_inv_21 = -lambda2 ./ g_det;   g_inv_22 = lambda1 ./ g_det;             
	
	phi = g_inv_21 .* lambda1 - g_inv_22 .* lambda2;
	xi     = 2 * phi + lambda3 * ones(np,1);
	tau    = delta_x' * ( delta_x ./ xi );		
	zeta   = 2 * vec( A * reshape( delta_x ./ xi, n, p ) );		
	
	Psi_blocks = cell(p,1);
	Psi_inv_blocks = cell(p,1);
	for ii = 1:p,
		Psi_blocks{ii} = - A * inv(diag(xi(n*(ii-1)+1:n*ii))) * A';
		Psi_inv_blocks{ii} = inv(Psi_blocks{ii});
	end       		
	
	Psi_inv_zeta = zeros(mp,1);
	for ii = 1:p,
		Psi_inv_zeta(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * zeta(m*(ii-1)+1:m*ii);
	end
	
	% V = kron(X,eye(m)) * Psi_inv * kron(X',eye(m));
	V = zeros(mn,mn);
	for ii = 1:n,
		for jj = ii:n,
			for kk = 1:p,
				V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj) = V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj) + (X(ii,kk) * X(jj,kk)) * Psi_inv_blocks{kk};
			end
			V(m*(jj-1)+1:m*jj,m*(ii-1)+1:m*ii) = V(m*(ii-1)+1:m*ii,m*(jj-1)+1:m*jj);
		end
	end		
	
	H = zeros(mn+n+1,mn+n+1);
	
	H(1:mn,1:mn) = lambda3 * eye(mn) - V; 
	H(1:mn,mn+1) = 2 * delta_a + vec( reshape( Psi_inv_zeta, m, p ) * X' ); 
	H(1:mn,mn+2:mn+n+1) = C_A; 
	
	H(mn+1,1:mn) = -2 * lambda3 * delta_a' - lambda3 * vec(reshape(Psi_inv_zeta,m,p) * X')';
	H(mn+1,mn+1) = -f3 + 4*lambda3*tau + lambda3*zeta'*Psi_inv_zeta;
	
	H(mn+2:mn+n+1,1:mn) = C_A';
	
	r_d1 = r_dual(1:np);
	r_d2 = r_dual(np+1:np+nm);
	r_d3 = r_dual(np+nm+1:2*np+nm);
	
	r_c1 = r_central(1:np);
	r_c2 = r_central(np+1:2*np);
	r_c3 = r_central(2*np+1);
	
	r_p1 = r_primal(1:mp);
	r_p2 = r_primal(mp+1:mp+n);
	
	u1 = -g_inv_11 .* r_c1 + g_inv_12 .* ( f2 .* r_d3 - r_c2 );
	u2 = -g_inv_21 .* r_c1 + g_inv_22 .* ( f2 .* r_d3 - r_c2 );	
			
	v = r_d3 - r_d1 - 2 * u2;
	
	y1 = -r_d2;
	y2 = -r_c3 + 2 * lambda3 * delta_x' * ( v ./ xi );
	y3 = -r_p1 - vec( A * reshape( v ./ xi, n, p  ));
	y4 = -r_p2;		
	
	Psi_inv_y3 = zeros(mp,1);
	for ii = 1:p,
		Psi_inv_y3(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * y3(m*(ii-1)+1:m*ii);
	end		
	
	z1 = y1 - vec( reshape(Psi_inv_y3,m,p) * X' );
	z2 = y2 - lambda3 * zeta' * Psi_inv_y3;
	z3 = y4;
	
	q = inv(H) * [ z1; z2; z3 ];
	
	delta_da = q(1:mn);
	delta_lambda3 = q(mn+1);
	delta_nu2 = q(mn+2:mn+n+1);		
	
	h = delta_lambda3 * zeta - vec( reshape(delta_da,m,n) * X ) + y3;
	
	delta_nu1 = zeros(mp,1);
	for ii = 1:p,
		delta_nu1(m*(ii-1)+1:m*ii) = Psi_inv_blocks{ii} * h(m*(ii-1)+1:m*ii);
	end		
	
	delta_dx = ( -2 * delta_lambda3 * delta_x - vec(A'* reshape(delta_nu1,m,p)) + v ) ./ xi;		
	delta_lambda1 = phi .* delta_dx + u2;
	delta_lambda2 = r_d3 - delta_lambda1;
	delta_w = ( g_inv_11 .* lambda1 - g_inv_12 .* lambda2 ) .* delta_dx + u1;	
	
	delta_x_pd = [delta_dx; delta_da; delta_w; delta_lambda1; delta_lambda2; delta_lambda3; delta_nu1; delta_nu2 ];
		
end