function [z,res_p,res_d] = sunsal_simple(M,y,lambda,err)

%% [x] = sunsal_v2(M,y,varargin)
%
%  SUNSAL -> sparse unmixing via variable splitting and augmented
%  Lagrangian methods 
%
%% --------------- Description --------------------------------------------
%
%  SUNSAL solves the following l2-l1 optimization  problem 
%  [size(M) = (L,p); size(X) = (p,N)]; size(Y) = (L,N)]
%
%         min  (1/2) ||M X-y||^2_F + lambda ||X||_1
%          X              
%
%  where ||X||_1 = sum(sum(abs(X)).
% 
%    CONSTRAINTS ACCEPTED:
%
%    1) POSITIVITY:  X >= 0;
%    2) ADDONE:  sum(X) = ones(1,N);
%
%    NOTES: 
%       1) The optimization w.r.t each column of X is decoupled. Thus, 
%          SUNSAL solves N simultaneous problems.
%
%       2) SUNSAL solves the following  problems:
%  
%          a) BPDN - Basis pursuit denoising l2-l1 
%                    (lambda > 0, POSITIVITY = 'no', ADDONE, 'no')
%
%          b) CBPDN - Constrained basis pursuit denoising l2-l1 
%                    (lambda > 0, POSITIVITY = 'yes', ADDONE, 'no')
%      
%          c) CLS   - Constrained least squares
%                     (lambda = 0, POSITIVITY = 'yes', ADDONE, 'no')
%
%          c) FCLS   - Fully constrained least squares
%                     (lambda >=0 , POSITIVITY = 'yes', ADDONE, 'yes')
%                      In this case, the regularizer ||X||_1  plays no role, 
%                      as it is constant.
%          
%
%% -------------------- Line of Attack  -----------------------------------
%
%  SUNSAL solves the above optimization problem by introducing a variable
%  splitting and then solving the resulting constrained optimization with
%  the augmented Lagrangian method of multipliers (ADMM). 
% 
% 
%         min  (1/2) ||M X-y||^2_F + lambda ||Z||_1
%          X,Z              
%         subject to: sum(X) = ones(1,N)); Z >= 0; X = Z
%
%  Augmented Lagrangian (scaled version):
%
%       L(X,Z,D) = (1/2) ||M X-y||^2_F + lambda ||Z||_1 + mu/2||X-Z-D||^2_F
%       
%  where D are the scale Lagrange multipliers
%
%
%  ADMM:
%
%      do 
%        X  <-- arg min L(X,Z,D)
%                    X, s.t: sum(X) = ones(1,N));
%        Z  <-- arg min L(X,Z,D)
%                    Z, s.t: Z >= 0;
%        D  <-- D - (X-Z);
%      while ~stop_rulde
%  
%For details see
%
%
% [1] J. Bioucas-Dias and M. Figueiredo, “Alternating direction algorithms
% for constrained sparse regression: Application to hyperspectral unmixing”, 
% in 2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal 
% Processing-WHISPERS'2010, Raykjavik, Iceland, 2010. 
%
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%
%  M - [L(channels) x p(endmembers)] mixing matrix
%
%  y - matrix with  L(channels) x N(pixels).
%      each pixel is a linear mixture of p endmembers
%      signatures y = M*x + noise,
%
%      
%
%
%%  ====================== Optional inputs =============================        
%  lambda - regularization parameter. lambda is either a scalar
%           or a vector with N components (one per column of x)
%           Default: 0. 
     
%%  =========================== Outputs ==================================
%
% X  =  [pxN] estimated mixing matrix
%
%

%%
% ------------------------------------------------------------------
% Author: Jose Bioucas-Dias, 2009
%
%

%
%% -------------------------------------------------------------------------
%
% Copyright (July, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
%
% SUNSAL is distributed under the terms of
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
% ---------------------------------------------------------------------

% mixing matrixsize
[LM,p] = size(M);
% data set size
[L,N] = size(y);
if (LM ~= L)
    error('mixing matrix M and data set y are inconsistent');
end
% if (L<p)
%     error('Insufficient number of columns in y');
% end
%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
% maximum number of AL iteration
AL_iters = 100;
% tolerance for the primal and dual residues
tol = err;
%%
%--------------------------------------------------------------
% Local variables
%--------------------------------------------------------------

%---------------------------------------------
%  If lambda is scalar convert it into vector
%---------------------------------------------
%lambda = lambda*ones(p,N);

% compute mean norm
norm_y = sqrt(mean(mean(abs(y).^2)));
% rescale M and Y and lambda
M = M/norm_y;
y = y/norm_y;
lambda = lambda/norm_y^2;


%%
%---------------------------------------------
%  Constants and initializations
%---------------------------------------------
mu_AL = 0.01;
mu = 10*mean(lambda(:)) + mu_AL;

% F = M'*M+mu*eye(p);
[UF,SF] = svd(M'*M);
sF = diag(SF);
IF = UF*diag(1./(sF+mu))*UF';
% IF = eye(p)/mu-M'*(eye(LM)+M*M'/mu)^(-1)*M/(mu*mu);
% IF = inv(F);
% Aux = IF*B'*inv(B*IF*B');


yy = M'*y;

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------

% no intial solution supplied
x= IF*M'*y;

z = x;
% scaled Lagrange Multipliers
d  = 0*z;


%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol1 = sqrt(N*p)*tol;
tol2 = sqrt(N*p)*tol;
i=1;
res_p = inf;
res_d = inf;
mu_changed = 0;
while (i <= AL_iters) && ((abs (res_p) > tol1) || (abs (res_d) > tol2)) 
    % save z to be used later
    if mod(i,10) == 1
        z0 = z;
    end
    % minimize with respect to z
    z = max(0,soft(x-d,lambda/mu));
    x = IF*(yy+mu*(z+d));
    
    % Lagrange multipliers update
    d = d -(x-z);

    % update mu so to keep primal and dual residuals whithin a factor of 10
    if mod(i,10) == 1
        % primal residue
%         res_p = norm(x-z,'fro');
        res_p = norm(abs(x-z),'fro');
        % dual residue
        %res_d = mu*norm(z-z0,'fro');
        res_d = mu*norm(abs(z-z0),'fro');
        % update mu
        if res_p > 10*res_d
            mu = mu*2;
            d = d/2;
            mu_changed = 1;
        elseif res_d > 10*res_p
            mu = mu/2;
            d = d*2;
            mu_changed = 1;
        end
        if  mu_changed
           IF = UF*diag(1./(sF+mu))*UF';
%            IF = eye(p)/mu-M'*(eye(LM)+M*M'/mu)^(-1)*M/(mu*mu);
%            Aux = IF*B'*inv(B*IF*B');
            mu_changed = 0;
        end
    end
    
    i=i+1;
       
end
 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
