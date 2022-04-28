function [M,Up,my,sing_values] = mvsa(Y,p,varargin)

%% [M,Up,my,sing_values] = mvsa(Y,p,varargin)
%
%  Minimum Volume Simplex Analysis (MVSA)
%
%% --------------- Description ---------------------------------------------
%
%  MVSA Estimates the vertices  M={m_1,...m_p} of the (p-1)-dimensional
%  simplex of minimum volume containing the vectors [y_1,...y_N], under the
%  assumption that y_i belongs to a (p-1)  dimensional affine set. Thus,
%  any vector y_i   belongs  to the convex hull of  the columns of M; i.e.,
%
%                   y_i = M*x_i
%
%  where x_i belongs to the probability (p-1)-simplex.
%
%  As described in the papers [1], [2], matrix M is  obtained by implementing
%  the following steps:
%
%   1-Project y onto a p-dimensional subspace containing the data set y
%
%            yp = Up'*y;      Up is an isometric matrix (Up'*Up=Ip)
%
%   2- solve the   optimization problem
%
%     Q^* = arg min_Q  -\log abs(det(Q))
%
%      subject to: Q*yp >= 0 and ones(1,p)*Q=mq,
%
%     where mq = ones(1,N)*yp'inv(yp*yp)
%
%   3- Compute
%
%      M = Up*inv(Q^*);
%
%% -------------------- Line of Attack  -----------------------------------
%
%  MVSA implements a  sequence of quadratic constrained subproblems. At
%  each iteration the gradient of the objective function is replaced
%  by a quadratic approximation followed by a line-search type procedure.
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%
% y - matrix with  L(channels) x N(pixels).
%     each pixel is a linear mixture of p endmembers
%     signatures y = M*x + noise,
%
%     MVSA assumes that y belongs to an affine space. It may happen,
%     however, that the data supplied by the user is not in an affine
%     set. For this reason, the first step this code implements
%     is the estimation of the affine set the best represent
%     (in the l2 sense) the data.
%
%  p - number of independent columns of M. Therefore, M spans a
%  (p-1)-dimensional affine set.
%
%
%%  ====================== Optional inputs =============================
%
%  'MM_ITERS' = double; Default 4;
%
%               Maximum number of constrained quadratic programs
%
%
%  'spherize'  = {'yes', 'no'}; Default 'yes'
%
%              Applies a spherization step to data such that the spherized
%              data spans over the same range along any axis.
%
%
%  'MU' = double; Default; 1e-6
%
%               Maximum eigenvalue of the quadratic approximation term
%
%
%  'LAMBDA' = double; Default; 1e-10
%
%               Spherization regularization parameter
%
%  'TOLF'  = double; Default; 1e-2
%
%              Tolerance for the termination test (relative variation of f(Q))
%
%
%
%  'M0'  =  <[Lxp] double>; Given by the VCA algorithm
%
%            Initial M.
%
%
%  'verbose'   = {0,1,2}; Default 1
%
%                 0 - work silently
%                 1 - display MVSA warnings
%                 2 - display MVSA and MATLAB warnings
%
%
%
%
%%  =========================== Outputs ==================================
%
% M  =  [Lxp] estimated mixing matrix
%
% Up =  [Lxp] isometric matrix spanning  the same subspace as M
%
% my =   mean value of y
%
% sing_values  = (p-1) eigenvalues of Cy = (y-my)*(y-my)/N. The dynamic range
%                  of these eigenvalues gives an idea of the  difficulty of the
%                  underlying problem
%
%
% NOTE: the identified affine set is given by
%
%              {z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}
%
%% -------------------------------------------------------------------------
%
% Copyright (May, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
%                               Jun li (jun@lx.it.pt)
%
% MVSA is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
% More details in:
%
% [1] Jun Li and José M. Bioucas-Dias
%     "Minimum volume simplex analysis: A fast algorithm to unmix hyperspectral data"
%      in IEEE International Geoscience and Remote sensing Symposium IGARSS’2008, Boston, USA,  2008
%
% [2] Jun Li and José M. Bioucas-Dias
%     "Minimum volume simplex analysis: A new algorithm to unmix
%     hyperspectral Data", IEEE Transactions on Geoscience and Remote
%     Sensing, 2009 (submitted).
%
% -------------------------------------------------------------------------

%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% data set size
[L,N] = size(Y);
if (L<p)
    error('Insufficient number of columns in y');
end
%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
% maximum number of quadratic QPs
MMiters = 4;
spherize = 'yes';
% display only MVSA warnings
verbose = 1;
% spherization regularization parameter
lambda = 1e-10;
% quadractic regularization parameter for the Hesssian
% Hreg = = mu*I+H
mu = 1e-6;
% no initial simplex
M = 0;
% tolerance for the termination test
tol_f = 1e-2;

%%
%--------------------------------------------------------------
% Local variables
%--------------------------------------------------------------
% maximum violation of inequalities
slack = 1e-3;
% flag energy decreasing
any_energy_decreasing = 0;
% used in the termination test
f_val_back = inf;
%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MM_ITERS'
                MMiters = varargin{i+1};
            case 'SPHERIZE'
                spherize = varargin{i+1};
            case 'MU'
                mu = varargin{i+1};
            case  'LAMBDA'
                lambda = varargin{i+1};
            case 'TOLF'
                tol_f = varargin{i+1};
            case 'M0'
                M = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

%%
%--------------------------------------------------------------
% set display mode
%--------------------------------------------------------------
if (verbose == 0) | (verbose == 1)
    warning('off','all');
    options = optimset('Display', 'off','Diagnostics','off','MaxIter',2000,'TolFun',1e-10,'TolX',1e-10 );
else
    warning('on','all');
    options = optimset('Display', 'final','Diagnostics','on','MaxIter',2000,'TolFun',1e-10,'TolX',1e-10);
end

%%
%--------------------------------------------------------------
% identify the affine space that best represent the data set y
%--------------------------------------------------------------
my = mean(Y,2);
Y = Y-repmat(my,1,N);
[Up,D] = svds(Y*Y'/N,p-1);
% represent y in the subspace R^(p-1)
Y = Up*Up'*Y;
% lift y
Y = Y + repmat(my,1,N);   %
% compute the orthogonal component of my
my_ortho = my-Up*Up'*my;
% define another orthonormal direction
Up = [Up my_ortho/sqrt(sum(my_ortho.^2))];
sing_values = diag(D);

% get coordinates in R^p
Y = Up'*Y;


%%
%------------------------------------------
% spherize if requested
%------------------------------------------
if strcmp(spherize,'yes')
    Y = Up*Y;
    Y = Y-repmat(my,1,N);
    C = diag(1./sqrt((diag(D+lambda*eye(p-1)))));
    Y=C*Up(:,1:p-1)'*Y;
    %  lift
    Y(p,:) = 1;
    % normalize to unit norm
    Y = Y/sqrt(p);
end


%---------------------------------------------
%  Initialization
%---------------------------------------------
if M == 0
    % Initialize with VCA
    Mvca = VCA(Y,'Endmembers',p);
    M = Mvca;
    % expand Q
    Ym = mean(M,2);
    Ym = repmat(Ym,1,p);
    dQ = M - Ym;
    % fraction: multiply by p is to make sure Q0 starts with a feasible
    % initial value.
    M = M + p*dQ;
else
    % Ensure that M is in the affine set defined by the data
    M = M-repmat(my,1,p);
    M = Up(:,1:p-1)*Up(:,1:p-1)'*M;
    M = M +  repmat(my,1,p);
    M = Up'*M;   % represent in the data subspace
    % is sherization is set
    if strcmp(spherize,'yes')
        M = Up*M-repmat(my,1,p);
        M=C*Up(:,1:p-1)'*M;
        %  lift
        M(p,:) = 1;
        % normalize to unit norm
        M = M/sqrt(p);
    end
end
Q0 = inv(M);
Q=Q0;


%%
%---------------------------------------------
%  build constraint matrices
%---------------------------------------------
% inequality matrix
A = kron(Y',eye(p));   % size np * p^2
% equality matrx
E = kron(eye(p),ones(1,p));
% equality independent vector
qm = sum(inv(Y*Y')*Y,2);


%%
%---------------------------------------------
%  sequence of QPs - main body
%---------------------------------------------
for k = 1:MMiters
    % make initial point feasible
    M = inv(Q);
    Ym = mean(M,2);
    Ym = repmat(Ym,1,p);
    dW = M - Ym;
    count = 0;
    while sum(sum( (inv(M)*Y) < 0  )) > 0
        M = M + 0.01*dW;
        count = count + 1;
        if count > 100
            if verbose
              fprintf('\n could not make M feasible after 100 expansions\n')
            end
            break;
        end
    end
    Q = inv(M);
    % gradient of -log(abs(det(Q)))
    g = -M';
    g = g(:);

    % quadractic term (mu*I+diag(H))
    H = mu*eye(p^2)+diag(g.^2);
    q0 = Q(:);
    Q0 = Q;
    f=g-H*q0;

    % initial function values (true and quadratic)
    f0_val = -log(abs(det(Q0)));
    f0_quad = f0_val; % (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);

    % anergy decreasing in this quadratic problem
    energy_decreasing = 0;
    
    %%%%%%%%%%%%%% QP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [q,f_dummy,exitflag, output] = quadprog(H,f,-A,zeros(p*N,1),E,qm,[],[],q0,options);

    if  exitflag < 0 % quadprog did not converge
        if verbose
            fprintf('\n iter = %d, quadprog did not converge: exitflag = %d \n', k, exitflag);
        end
        if k == MMiters
           if ~any_energy_decreasing
               if verbose
                   fprintf('\n outputing the VCA solution\n');
               end
               Q = inv(Mvca);
               q = Q(:);
           else
               if verbose
                   fprintf('\n outputing the solution of the previous iteration\n');
               end
              Q = Q0;
              q = Q(:);
           end
        else
            % run again with a larger mu
            Q = Q0;
            q = Q(:);
            mu = 1e-2;
        end
    elseif exitflag == 0 % Number of iterations exceeded options.MaxIter.
        % compute  energy of the quadratic approximation
        f_quad = f0_val + (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);
        if verbose
            fprintf('\n iterations exceeded: iter = %d, f0_quad = %2.4f, f_quad = %2.4f, iter(QP) = %d \n',...
                k, f0_quad,f_quad,output.iterations)
        end
        %test for energy decreasing and feasibility
        if (f0_quad > f_quad) & (sum(sum( Q*Y < -slack  )) == 0)
            if verbose
                fprintf('\n test for quadratic energy decreasing and feasibility PASSED\n')
            end
            % there will be surely an energy decreasing between for Q
            % between the current Q and Q0
            energy_decreasing = 1;
            any_energy_decreasing = 1;
        else
            if verbose
                fprintf('\n test for quadratic energy decreasing FAILED\n')
                fprintf('\n Incremente H\n')
            end
            % increment H
            Q = Q0;
            q = Q(:);
            mu = 1e-2;
        end
    end

    % energy_decreasing == 1 means that  although exitflag ~= 1, the
    % energy of the quadratic approximation decreased.
    % exiflaf == 1 implies that the energy of the quadratic approximation
    % decreased.
    if energy_decreasing | (exitflag == 1)
        Q = reshape(q,p,p);
        %f_bound
        f_val = -log(abs(det(Q)));
        if verbose
            fprintf('\n iter = %d, f0 = %2.4f, f = %2.4f, exitflag = %d, iter(QP) = %d \n',...
                k, f0_val,f_val,exitflag,output.iterations)
        end
        % line search
        counts = 1;
        while (f0_val < f_val)
            % Q and Q0 are in a convex set and f(alpha*Q0+(1-alpha)Q) <
            % f(Q0)  for some alpha close to zero
            Q = (Q+Q0)/2;
            f_val = -log(abs(det(Q)));
            if verbose
                fprintf('\n doing line search: counts = %d, f0 = %2.4f, f = %2.4f\n', ...
                    counts, f0_val, f_val)
            end
            counts = counts + 1;
            if counts > 20
                fprintf('\n something wrong with the line search\n')
                if k == MMiters
                    if ~energy_decreasing
                        fprintf('\n outputing the VCA solution\n');
                        Q = inv(Mvca);
                    else
                        fprintf('\n outputing the solution of the previous iteration\n');
                        Q = Q0;
                    end
                else
                    % run again with a larger mu
                    Q = Q0;
                    mu = 1e-2;
                end
            end
        end
        energy_decreasing = 1;
    end
    % termination test
    if energy_decreasing
        if abs((f_val_back-f_val)/f_val) < tol_f
            if verbose
                fprintf('\n iter: = %d termination test PASSED \n', k)
            end
            break;
        end
        f_val_back = f_val;
    end
end


if strcmp(spherize,'yes')
    M = inv(Q);
    % refer to the initial affine set
    % unscale
    M = M*sqrt(p);
    %remove offset
    M = M(1:p-1,:);
    % unspherize
    M = Up(:,1:p-1)*diag(sqrt(diag(D+lambda*eye(p-1))))*M;
    % sum ym
    M = M + repmat(my,1,p);
else
    M = Up*inv(Q);
end



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % 