function [M,Up,my,sing_values] = sisal(Y,p,varargin)

%% [M,Up,my,sing_values] = sisal(Y,p,varargin)
%
% Simplex identification via split augmented Lagrangian (SISAL)
%
%% --------------- Description ---------------------------------------------
%
%  SISAL Estimates the vertices  M={m_1,...m_p} of the (p-1)-dimensional
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
%       Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
%
%                 subject to:  ones(1,p)*Q=mq,
%
%      where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
%              induced norm (see [1])
%   3- Compute
%
%      M = Up*inv(Q^*);
%
%% -------------------- Line of Attack  -----------------------------------
%
% SISAL replaces the usual fractional abundance positivity constraints, 
% forcing the spectral vectors to belong to the convex hull of the 
% endmember signatures,  by soft  constraints. This new criterion brings
% robustnes to noise and outliers
%
% The obtained optimization problem is solved by a sequence of
% augmented Lagrangian optimizations involving quadractic and one-sided soft
% thresholding steps. The resulting algorithm is very fast and able so
% solve problems far beyond the reach of the current state-of-the art
% algorithms. As examples, in a standard PC, SISAL, approximatelly, the
% times:
%
%  p = 10, N = 1000 ==> time = 2 seconds
%
%  p = 20, N = 50000 ==> time = 3 minutes
%
%%  ===== Required inputs =============
%
% y - matrix with  L(channels) x N(pixels).
%     each pixel is a linear mixture of p endmembers
%     signatures y = M*x + noise,
%
%     SISAL assumes that y belongs to an affine space. It may happen,
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
%  'MM_ITERS' = double; Default 80;
%
%               Maximum number of constrained quadratic programs
%
%
%  'TAU' = double; Default; 1
%
%               Regularization parameter in the problem
%
%               Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
%
%                 subject to:ones(1,p)*Q=mq,
%
%              where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
%              induced norm (see [1]).
%
%  'MU' = double; Default; 1
%
%              Augmented Lagrange regularization parameter
%
%  'spherize'  = {'yes', 'no'}; Default 'yes'
%
%              Applies a spherization step to data such that the spherized
%              data spans over the same range along any axis.
%
%  'TOLF'  = double; Default; 1e-2
%
%              Tolerance for the termination test (relative variation of f(Q))
%
%
%  'M0'  =  <[Lxp] double>; Given by the VCA algorithm
%
%            Initial M.
%
%
%  'verbose'   = {0,1,2,3}; Default 1
%
%                 0 - work silently
%                 1 - display simplex volume
%                 2 - display figures
%                 3 - display SISAL information 
%                 4 - display SISAL information and figures
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
%                 of these eigenvalues gives an idea of the  difficulty of the
%                 underlying problem
%
%
% NOTE: the identified affine set is given by
%
%              {z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}
%
%% -------------------------------------------------------------------------
%
% Copyright (May, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
%
% SISAL is distributed under the terms of
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
% [1] José M. Bioucas-Dias
%     "A variable splitting augmented lagrangian approach to linear spectral unmixing"
%      First IEEE GRSS Workshop on Hyperspectral Image and Signal
%      Processing - WHISPERS, 2009 (submitted). http://arxiv.org/abs/0904.4635v1
%
%
%
% -------------------------------------------------------------------------
%
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
MMiters = 80;
spherize = 'yes';
% display only volume evolution
verbose = 1;
% soft constraint regularization parameter
tau = 1;
% Augmented Lagrangian regularization parameter
mu = p*1000/N;
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
energy_decreasing = 0;
% used in the termination test
f_val_back = inf;
%
% spherization regularization parameter
lam_sphe = 1e-8;
% quadractic regularization parameter for the Hesssian
% Hreg = = mu*I
lam_quad = 1e-6;
% minimum number of AL iterations per quadratic problem 
AL_iters = 4;
% flag 
flaged = 0;

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
            case  'TAU'
                tau = varargin{i+1};
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
if (verbose == 3) | (verbose == 4)
    warning('off','all');
else
    warning('on','all');
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
    C = diag(1./sqrt((diag(D+lam_sphe*eye(p-1)))));
    IC = inv(C);
    Y=C*Up(:,1:p-1)'*Y;
    %  lift
    Y(p,:) = 1;
    % normalize to unit norm
    Y = Y/sqrt(p);
end

%%
% ---------------------------------------------
%            Initialization
%---------------------------------------------
if M == 0
    % Initialize with VCA
    Mvca = VCA(Y,'Endmembers',p,'verbose','off');
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


% plot  initial matrix M
if verbose == 2 | verbose == 4
    set(0,'Units','pixels')

    %get figure 1 handler
    H_1=figure;
    pos1 = get(H_1,'Position');
    pos1(1)=50;
    pos1(2)=100+400;
    set(H_1,'Position', pos1)

    hold on
    M = inv(Q);
    p_H(1) = plot(Y(1,:),Y(2,:),'.');
    p_H(2) = plot(M(1,:), M(2,:),'ok');

    leg_cell = cell(1);
    leg_cell{1} = 'data points';
    leg_cell{end+1} = 'M(0)';
    title('SISAL: Endmember Evolution')

end


%%
% ---------------------------------------------
%            Build constant matrices
%---------------------------------------------

AAT = kron(Y*Y',eye(p));    % size p^2xp^2
B = kron(eye(p),ones(1,p)); % size pxp^2
qm = sum(inv(Y*Y')*Y,2);


H = lam_quad*eye(p^2);
F = H+mu*AAT;          % equation (11) of [1]
IF = inv(F);

% auxiliar constant matrices
G = IF*B'*inv(B*IF*B');
qm_aux = G*qm;
G = IF-G*B*IF;


%%
% ---------------------------------------------------------------
%          Main body- sequence of quadratic-hinge subproblems
%----------------------------------------------------------------

% initializations
Z = Q*Y;
Bk = 0*Z;


for k = 1:MMiters
    
    IQ = inv(Q);
    g = -IQ';
    g = g(:);

    baux = H*Q(:)-g;

    q0 = Q(:);
    Q0 = Q;
    
    % display the simplex volume
    if verbose == 1
        if strcmp(spherize,'yes')
            % unscale
            M = IQ*sqrt(p);
            %remove offset
            M = M(1:p-1,:);
            % unspherize
            M = Up(:,1:p-1)*IC*M;
            % sum ym
            M = M + repmat(my,1,p);
            M = Up'*M;
        else
            M = IQ;
        end
        fprintf('\n iter = %d, simplex volume = %4f  \n', k, 1/abs(det(M)))
    end

    
    %Bk = 0*Z;
    if k==MMiters
        AL_iters = 100;
        %Z=Q*Y;
        %Bk = 0*Z;
    end
    
    % initial function values (true and quadratic)
    % f0_val = -log(abs(det(Q0)))+ tau*sum(sum(hinge(Q0*Y)));
    % f0_quad = f0_val; % (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);
    
    while 1 > 0
        q = Q(:);
        % initial function values (true and quadratic)
        f0_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
        f0_quad = (q-q0)'*g+1/2*(q-q0)'*H*(q-q0) + tau*sum(sum(hinge(Q*Y)));
        for i=2:AL_iters
            %-------------------------------------------
            % solve quadratic problem with constraints
            %-------------------------------------------
            dq_aux= Z+Bk;             % matrix form
            dtz_b = dq_aux*Y';
            dtz_b = dtz_b(:);
            b = baux+mu*dtz_b;        % (11) of [1]
            q = G*b+qm_aux;           % (10) of [1]
            Q = reshape(q,p,p);
            
            %-------------------------------------------
            % solve hinge
            %-------------------------------------------
            Z = soft_neg(Q*Y -Bk,tau/mu);
            
                 %norm(B*q-qm)
           
            %-------------------------------------------
            % update Bk
            %-------------------------------------------
            Bk = Bk - (Q*Y-Z);
            if verbose == 3 ||  verbose == 4
                fprintf('\n ||Q*Y-Z|| = %4f \n',norm(Q*Y-Z,'fro'))
            end
            if verbose == 2 || verbose == 4
                M = inv(Q);
                plot(M(1,:), M(2,:),'.r');
                if ~flaged
                     p_H(3) = plot(M(1,:), M(2,:),'.r');
                     leg_cell{end+1} = 'M(k)';
                     flaged = 1;
                end
            end
        end
        f_quad = (q-q0)'*g+1/2*(q-q0)'*H*(q-q0) + tau*sum(sum(hinge(Q*Y)));
        if verbose == 3 ||  verbose == 4
            fprintf('\n MMiter = %d, AL_iter, = % d,  f0 = %2.4f, f_quad = %2.4f,  \n',...
                k,i, f0_quad,f_quad)
        end
        f_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
        if f0_quad >= f_quad    %quadratic energy decreased
            while  f0_val < f_val;
                if verbose == 3 ||  verbose == 4
                    fprintf('\n line search, MMiter = %d, AL_iter, = % d,  f0 = %2.4f, f_val = %2.4f,  \n',...
                        k,i, f0_val,f_val)
                end
                % do line search
                Q = (Q+Q0)/2;
                f_val = -log(abs(det(Q)))+ tau*sum(sum(hinge(Q*Y)));
            end
            break
        end
    end



end

if verbose == 2 || verbose == 4
    p_H(4) = plot(M(1,:), M(2,:),'*g');
    leg_cell{end+1} = 'M(final)';
    legend(p_H', leg_cell);
end


if strcmp(spherize,'yes')
    M = inv(Q);
    % refer to the initial affine set
    % unscale
    M = M*sqrt(p);
    %remove offset
    M = M(1:p-1,:);
    % unspherize
    M = Up(:,1:p-1)*IC*M;
    % sum ym
    M = M + repmat(my,1,p);
else
    M = Up*inv(Q);
end


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %