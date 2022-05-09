function [deltaA,deltaX] = tangent_space_L1_ip_clamped(A,X,eps)

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
%     using the prototype algorithm of Boyd and Vandenberghe, Chapter 11,
%     with a few tricks to solve the Newton system more efficiently. This
%     version has difficulty with small step sizes when complementarity is
%     approached in a few of the variables. 
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

	VERBOSE        = 2;
    ALPHA          = 0.05;
    BETA           = 0.5;
    MU             = 10;
    MAX_ITERATIONS = 30; 
    MAX_LINE_SEARCH_ITERATIONS = 50;

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
    eval_r_central = @(z,tc) [ -get_lambda_1(z) .* eval_f1(z); -get_lambda_2(z) .* eval_f2(z); -get_lambda_3(z) * eval_f3(z) ] - [ ones(2*np,1) / tc; 100 / tc ];
    eval_r_primal  = @(z)[ eval_L1(z) ; eval_L2(z) ];
    eval_sdg = @(z) - eval_f1(z)' * get_lambda_1(z) - eval_f2(z)' * get_lambda_2(z) - eval_f3(z) * get_lambda_3(z);

    % initialization: 
    x_pd = zeros(totalVars,1);
	x_pd(np+nm+1:2*np+nm) = vec(1.5*abs(X));
	x_pd(2*np+nm+1:4*np+nm+1) = ones(2*np+1,1);

    eta_hat = eval_sdg(x_pd);

    numIterations = 0;
    converged = false;

    while ~converged
	
        numIterations = numIterations + 1;

        t = MU * ( 2 * n * p + 1 ) / eta_hat;       

        % set up the right hand side (calcluate residuals)
        r_dual     =  eval_r_dual(x_pd); 
        r_central  =  eval_r_central(x_pd,t);
        r_primal   =  eval_r_primal(x_pd);

		delta_x = get_delta_x(x_pd);
		delta_a = get_delta_a(x_pd);
		lambda1 = get_lambda_1(x_pd);
		lambda2 = get_lambda_2(x_pd);
		lambda3 = get_lambda_3(x_pd);
		f1 = eval_f1(x_pd);
		f2 = eval_f2(x_pd);
		f3 = eval_f3(x_pd);
		nu1 = get_nu_1(x_pd);
		nu2 = get_nu_2(x_pd);

        delta_x_pd = solve_step_equations(A,X,delta_x,delta_a,lambda1,lambda2,lambda3,f1,f2,f3,nu1,nu2,r_primal,r_central,r_dual);

        % step and backtracking line search 
        lineSearchConverged = 0;
        numLineSearchIterations = 0;

        delta_lambda1 = get_lambda_1(delta_x_pd);
        delta_lambda2 = get_lambda_2(delta_x_pd);
		delta_lambda3 = get_lambda_3(delta_x_pd);

        I1 = find(delta_lambda1 < 0);
        q1 = -lambda1(I1) ./ delta_lambda1(I1);
        I2 = find(delta_lambda2 < 0);
        q2 = -lambda2(I2) ./ delta_lambda2(I2); 
        
        f = [ f1; f2 ];        
        v1 =     get_delta_x(delta_x_pd) - get_w(delta_x_pd);
        v2 =   - get_delta_x(delta_x_pd) - get_w(delta_x_pd);
        v = [ v1; v2 ];        
        J = find(v > 0);        

        sMax = min([1, min(q1), min(q2), min(-f(J) ./ v(J))]);			    

		if delta_lambda3 < 0, 
			sMax = min(sMax,-lambda3 / delta_lambda3);
		end
		
		delta_dx = get_delta_x(delta_x_pd);
		delta_da = get_delta_a(delta_x_pd);
		
		a = norm( delta_dx )^2 + norm(delta_da)^2;
		b = 2 * ( delta_dx' * delta_x + delta_da' * delta_a );
		c = norm(delta_x)^2 + norm(delta_a)^2 - eps^2;	
		sMaxBall = (- b + sqrt(b^2-4*a*c)) / (2 * a);
				
		sMax = min(sMax,sMaxBall);		
        s = .99 * sMax;

        oldResidualNorm = sqrt( r_dual'*r_dual + r_central'*r_central + r_primal'*r_primal );

        while ~lineSearchConverged, 

            numLineSearchIterations = numLineSearchIterations + 1;

            x_pd_new = x_pd + s * delta_x_pd; 

            r_dual_new    = eval_r_dual(x_pd_new);
            r_central_new = eval_r_central(x_pd_new,t);
            r_primal_new  = eval_r_primal(x_pd_new);        

            newResidualNorm = sqrt( r_dual_new'*r_dual_new + r_central_new'*r_central_new + r_primal_new'*r_primal_new );

            if ( newResidualNorm <= (1-ALPHA*s)*oldResidualNorm || numLineSearchIterations >= MAX_LINE_SEARCH_ITERATIONS )

                lineSearchConverged = true;
                x_pd = x_pd_new;

                if VERBOSE > 2,
                    disp(['   Line search terminated at iteration: ' num2str(numLineSearchIterations) '   sMax: ' num2str(sMax) '   s: ' num2str(s) '   Prev res: ' num2str(oldResidualNorm) '  New res: ' num2str(newResidualNorm)]);
                end
            else
				%if VERBOSE > 3,
				%	disp(['   r_d: ' num2str(norm(r_dual)) ' ' num2str(norm(r_dual_new)) '   r_c: ' num2str(norm(r_central)) ' ' num2str(norm(r_central_new)) '   r_p: ' num2str(norm(r_primal)) ' ' num2str(norm(r_primal_new))]);
				%end
					
                s = BETA * s;
            end
        end

        % check convergence: duality gap or number of iterations
        eta_hat = eval_sdg(x_pd); 

        if VERBOSE > 1, 
            disp(['        Interior point iteration ' num2str(numIterations) '   SDG: ' num2str(eta_hat / (2*np+1)) '   PD Res: ' num2str(newResidualNorm) '   t: ' num2str(t) ]);
        end 

        converged = ( numIterations >= MAX_ITERATIONS ) || ((eta_hat / (2*np+1)) < 1e-7 );

    end     

    deltaA = reshape(get_delta_a(x_pd),m,n);
    deltaX = reshape(get_delta_x(x_pd),n,p);
    
    if VERBOSE > 0,
        disp(['        Interior point terminated with SDG: ' num2str(eta_hat) '   PD Residual: ' num2str(newResidualNorm)]);
    end
    
    return;

end

function delta_x_pd = solve_step_equations(A,X,delta_x,delta_a,lambda1,lambda2,lambda3,f1,f2,f3,nu1,nu2,r_primal,r_central,r_dual)

	REDUCTION = 4;

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
	% DIRECT INVERSION
	%%%
	
	if REDUCTION == 0,
		H = zeros(totalVars,totalVars);
	
		H(1:np,1:np)                   =  lambda3 * eye(np);
		H(1:np,2*np+nm+1:3*np+nm)      =  eye(np);
		H(1:np,3*np+nm+1:4*np+nm)      = -eye(np);
		H(1:np,4*np+nm+1)              = 2*delta_x;
		H(1:np,4*np+nm+2:4*np+nm+mp+1) = kron(eye(p),A');
		
		H(np+1:np+nm,np+1:np+nm)                  = lambda3 * eye(mn);
		H(np+1:np+nm,4*np+nm+1)                   = 2*delta_a;
		H(np+1:np+nm,4*np+nm+2:4*np+nm+mp+1)      = kron(X,eye(m));
		H(np+1:np+nm,4*np+nm+mp+2:4*np+nm+mp+n+1) = C_A;
		
		H(np+nm+1:2*np+nm,2*np+nm+1:3*np+nm) = -eye(np);
		H(np+nm+1:2*np+nm,3*np+nm+1:4*np+nm) = -eye(np);
		
		H(2*np+nm+1:3*np+nm,1:np)              = -diag(lambda1);
		H(2*np+nm+1:3*np+nm,np+nm+1:2*np+nm)   =  diag(lambda1);
		H(2*np+nm+1:3*np+nm,2*np+nm+1:3*np+nm) = -diag(f1);
		
		H(3*np+nm+1:4*np+nm,1:np)              = diag(lambda2);
		H(3*np+nm+1:4*np+nm,np+nm+1:2*np+nm)   = diag(lambda2);
		H(3*np+nm+1:4*np+nm,3*np+nm+1:4*np+nm) = -diag(f2); 
		
		H(4*np+nm+1,1:np)         = -2*lambda3*delta_x';
		H(4*np+nm+1,np+1:np+nm)   = -2*lambda3*delta_a';
		H(4*np+nm+1,4*np+nm+1)    = -f3;

		H(4*np+nm+2:4*np+nm+mp+1,1:np)       = kron(eye(p),A);
		H(4*np+nm+2:4*np+nm+mp+1,np+1:np+nm) = kron(X',eye(m));
		
		H(4*np+nm+mp+2:4*np+nm+mp+n+1,np+1:np+nm) = C_A';
		
		delta_x_pd = inv(H) * [ -r_dual; -r_central; -r_primal ]; 
	
		return; 
	end
	
	if REDUCTION == 1,
	
		%%%
		%   Solve (6.67) and back substitute
		%%%
		
		% invert gamma
		g_det = lambda1 .* f2 + lambda2 .* f1;        
		g_inv_11 =       f2 ./ g_det;   g_inv_12 =      f1 ./ g_det;
		g_inv_21 = -lambda2 ./ g_det;   g_inv_22 = lambda1 ./ g_det;             
		
		phi = g_inv_21 .* lambda1 - g_inv_22 .* lambda2;
		
		H = zeros(np+nm+1+mn+n,np+nm+1+mn+n);
		
		H(1:np,1:np)               = 2 * diag(phi) + lambda3 * eye(np);
		H(1:np,np+nm+1)            = 2 * delta_x; 
		H(1:np,np+nm+2:np+nm+mp+1) = kron(eye(p),A');
		
		H(np+1:np+nm,np+1:np+nm) = lambda3 * eye(nm);
		H(np+1:np+nm,np+nm+1)    = 2 * delta_a;
		H(np+1:np+nm,np+nm+2:np+nm+mp+1) = kron(X,eye(m));
		H(np+1:np+nm,np+nm+mp+2:np+nm+mp+n+1) = C_A;
		
		H(np+nm+1,1:np) = - 2* lambda3 * delta_x';
		H(np+nm+1,np+1:np+nm) = -2 * lambda3 * delta_a';
		H(np+nm+1,np+nm+1) = -f3;
		
		H(np+nm+2:np+nm+mp+1,1:np) = kron(eye(p),A);
		H(np+nm+2:np+nm+mp+1,np+1:np+nm) = kron(X',eye(m));
			
		H(np+nm+mp+2:np+nm+mp+n+1,np+1:np+nm) = C_A';
		
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
		
		rhs = [ r_d3 - r_d1 - 2 * u2; -r_d2; -r_c3; -r_p1; -r_p2 ];
		
		q = inv(H) * rhs;
		
		delta_dx = q(1:np);
		delta_da = q(np+1:np+nm);
		delta_lambda3 = q(np+nm+1);
		delta_nu1 = q(np+nm+2:np+nm+mp+1);
		delta_nu2 = q(np+nm+mp+2:np+nm+mp+n+1);
		
		delta_lambda1 = phi .* delta_dx + u2;
		delta_lambda2 = r_d3 - delta_lambda1;
		delta_w = ( g_inv_11 .* lambda1 - g_inv_12 .* lambda2 ) .* delta_dx + u1;	
		
		delta_x_pd = [delta_dx; delta_da; delta_w; delta_lambda1; delta_lambda2; delta_lambda3; delta_nu1; delta_nu2 ];
		
		return;
	end;
		
	if REDUCTION == 2,
		%%%
		%   Solve (6.76) and back substitute
		%%%
		
		% invert gamma
		g_det = lambda1 .* f2 + lambda2 .* f1;        
		g_inv_11 =       f2 ./ g_det;   g_inv_12 =      f1 ./ g_det;
		g_inv_21 = -lambda2 ./ g_det;   g_inv_22 = lambda1 ./ g_det;             
		
		phi = g_inv_21 .* lambda1 - g_inv_22 .* lambda2;
				
		Xi = 2 * diag(phi) + lambda3 * eye(np);
		Xi_inv = inv(Xi);
		zeta = 2 * kron(eye(p),A) * Xi_inv * delta_x;		
		Psi = - kron(eye(p),A) * Xi_inv * kron(eye(p),A');
		tau = delta_x' * Xi_inv * delta_x; 
		
		H = zeros(mn+1+mp+n,mn+1+mp+n);
		
		H(1:mn,1:mn) = lambda3 * eye(mn);
		H(1:mn,mn+1) = 2 * delta_a;
		H(1:mn,mn+2:mn+mp+1) = kron(X,eye(m));
		H(1:mn,mn+mp+2:mn+mp+n+1) = C_A;
		
		H(mn+1,1:mn) = - 2 * lambda3 * delta_a';
		H(mn+1,mn+1) = -f3 + 4 * lambda3 * tau; 
		H(mn+1,mn+2:mn+mp+1) = lambda3 * zeta';
		
		H(mn+2:mn+mp+1,1:mn) = kron(X',eye(m));
		H(mn+2:mn+mp+1,mn+1) = -zeta;
		H(mn+2:mn+mp+1,mn+2:mn+mp+1) = Psi;
		
		H(mn+mp+2:mn+mp+n+1,1:mn) = C_A';
		
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
		y2 = -r_c3 + 2 * lambda3 * delta_x' * Xi_inv * v;
		y3 = -r_p1 - kron(eye(p),A) * Xi_inv * v;
		y4 = -r_p2;
		
		q = inv(H) * [ y1; y2; y3; y4 ];
	
		delta_da = q(1:mn);
		delta_lambda3 = q(mn+1);
		delta_nu1 = q(mn+2:mn+mp+1);
		delta_nu2 = q(mn+mp+2:mn+mp+n+1);
		
		delta_dx = Xi_inv * ( -2 * delta_lambda3 * delta_x - kron(eye(p),A') * delta_nu1 + v );
		
		delta_lambda1 = phi .* delta_dx + u2;
		delta_lambda2 = r_d3 - delta_lambda1;
		delta_w = ( g_inv_11 .* lambda1 - g_inv_12 .* lambda2 ) .* delta_dx + u1;	
		
		delta_x_pd = [delta_dx; delta_da; delta_w; delta_lambda1; delta_lambda2; delta_lambda3; delta_nu1; delta_nu2 ];

	end
	
	if REDUCTION == 3,
		
		%%%
		%   Solve using complete reduction to system of size mn+n+1
		%%%
		
		% invert gamma
		g_det    = lambda1 .* f2 + lambda2 .* f1;        
		g_inv_11 =       f2 ./ g_det;   g_inv_12 =      f1 ./ g_det;
		g_inv_21 = -lambda2 ./ g_det;   g_inv_22 = lambda1 ./ g_det;             
		
		phi = g_inv_21 .* lambda1 - g_inv_22 .* lambda2;
				
		Xi     = 2 * diag(phi) + lambda3 * eye(np);
		Xi_inv = inv(Xi);
		zeta   = 2 * kron(eye(p),A) * Xi_inv * delta_x;		
		Psi    = - kron(eye(p),A) * Xi_inv * kron(eye(p),A');
		tau    = delta_x' * Xi_inv * delta_x; 
		Psi_inv = inv(Psi);
		
		H = zeros(mn+n+1,mn+n+1);
		
		H(1:mn,1:mn) = lambda3 * eye(mn) - kron(X,eye(m)) * Psi_inv * kron(X',eye(m));
		H(1:mn,mn+1) = 2 * delta_a + kron(X,eye(m)) * Psi_inv * zeta;
		H(1:mn,mn+2:mn+n+1) = C_A; 
		
		H(mn+1,1:mn) = -2 * lambda3 * delta_a' - lambda3 * zeta' * Psi_inv * kron(X',eye(m));
		H(mn+1,mn+1) = -f3 + 4*lambda3*tau + lambda3*zeta'*Psi_inv*zeta;
		
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
		y2 = -r_c3 + 2 * lambda3 * delta_x' * Xi_inv * v;
		y3 = -r_p1 - kron(eye(p),A) * Xi_inv * v;
		y4 = -r_p2;		
		
		z1 = y1 - kron(X,eye(m)) * Psi_inv * y3;
		z2 = y2 - lambda3 * zeta' * Psi_inv * y3;
		z3 = y4;
		
		q = inv(H) * [ z1; z2; z3 ];
		
		delta_da = q(1:mn);
		delta_lambda3 = q(mn+1);
		delta_nu2 = q(mn+2:mn+n+1);		
		delta_nu1 = Psi_inv * ( delta_lambda3 * zeta - kron(X',eye(m)) * delta_da + y3 );		
		delta_dx = Xi_inv * ( -2 * delta_lambda3 * delta_x - kron(eye(p),A') * delta_nu1 + v );		
		delta_lambda1 = phi .* delta_dx + u2;
		delta_lambda2 = r_d3 - delta_lambda1;
		delta_w = ( g_inv_11 .* lambda1 - g_inv_12 .* lambda2 ) .* delta_dx + u1;	
		
		delta_x_pd = [delta_dx; delta_da; delta_w; delta_lambda1; delta_lambda2; delta_lambda3; delta_nu1; delta_nu2 ];
		
	end
	
	if REDUCTION == 4,
		
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
		
end