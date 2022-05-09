function [u, pnn] = RS_FGP_TV(data, alpha, niter, opts, pn)
%RS_FGP_TV Runs the fast gradient projection algorithm [1] on the AQPL dual problem.
%
% [1] Beck, A., & Teboulle, M. (2009). Fast gradient-based algorithms for 
%   constrained total variation image denoising and deblurring problems. 
%   IEEE Transactions on Image Processing, 18(11), 2419-2434. doi:10.1109/TIP.2009.2028250
%
% Input:    
%   data [matrix]              
%       noisy input image (in paper this is called f)
%   alpha [float]       
%       regularization parameter
%   niter [int]
%       maximum number of iterations.
%   opts [struct; optional]            
%       there are some options of the algorithm which can be changed   
%
%       opts.PC [function handle; DEFAULT = @(x) x]
%           projection mapping fo box constraints
%       opts.tol [scalar; DEFAULT = 0]
%           tolerance for convergence. The default does not check anything
%           as it slows done the computation
%       opts.min_iter [int; DEFAULT = 0]
%           minimum number of iterations that are required even if opts.tol is already
%           fulfilled
%   pn [matrix; DEFAULT = zeros]
%       initial guess for the dual state. I guess that any [NxMx2] matrix
%       whose norm over the third component is smaller than 1 can be taken
%       as an inital guess. It might be that any matrix can be used.
%    
% Output:
%   u [matrix]
%       denoised image
%   pnn [matrix]
%       dual state of the final iterate
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

% treat the input options
if ~exist('opts', 'var'), opts = struct; end;
if ~isfield(opts, 'PC'), opts.PC = @(x) x; end;
if ~isfield(opts, 'tol'), opts.tol = 0; end;
if ~isfield(opts, 'min_iter'), opts.min_iter = 0; end;

if nargin < 5; pn = zeros([size(data) 2]); end;

if alpha == 0
    u = data;
    pnn = pn;
    return
end

% initialize
factr = 1./(8*alpha);
q = pn;
pnn = pn;
t_n = 1;

for n = 1 : niter
    s = opts.PC(data + alpha* D(q));
    g = G(s);
    pnn = PP(q + factr * g);
    
    if opts.tol > 0 && n >= opts.min_iter
        dp = norm(pnn(:) - pn(:))/norm(pn(:));
        %             fprintf('%2.2e\n', dp);
        if dp < opts.tol
            %                 fprintf('iter: %d, tol reached for dTV\n', n);
            break
        end
    end
    
    if n == niter;
        break;
    end;
    
    t_nn = (1 + sqrt(1 + 4*t_n^2))./2;
    q = pnn + (t_n-1)./t_nn * (pnn - pn);
    
    pn = pnn;
    t_n = t_nn;
end

u = opts.PC(data + alpha * D(pnn));
end

function x = G(x)
x = RS_gradient(x);
end

function x = D(x)
x = RS_divergence(x);
end

function x = PP(x)
% pointwise projection onto unit ball
nx = max(1, sqrt(sum(x.^2, 3)));
x = bsxfun(@rdivide, x, nx);
end
