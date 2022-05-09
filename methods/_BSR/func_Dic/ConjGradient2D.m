function [x,flag,iter,resvec,func] = ConjGradient2D(A,b,varargin)
% - Conj Gradient for 2D data. 
% - input
% b   : known term
% A   : function handle to matrix vector product

% - output
% flag: returns the end condition
%       flag = 0 - end condition reached
%       flag = 1 - no convergence in maxit iterations
%       flag = 2 - numerical problems (see text message) 
% iter: number of iterations to reach convergence
% rvec: residual vector
% func: values of the quadratic function to minimize L(x)=1/2*x'Ax-b'x

% - preconditioning matrix
if nargin < 3 || isempty(varargin{1})
    M = []; else M = varargin{1}; end

% - tolerance for convergence
if nargin < 4 || isempty(varargin{2})
    tol = 1e-3; else tol = varargin{2}; end

% - maximum number of iterations
if nargin < 5 || isempty(varargin{3})
    maxit = 50; else maxit = varargin{3}; end 

% - initial guess
if nargin < 6 || isempty(varargin{4})
    x0 = zeros(size(b)); else x0 = varargin{4}; end 

% - stopping rule
if nargin < 7 || isempty(varargin{5})
    stopping_rule = 3; else stopping_rule = varargin{5}; end 

% - display hystory runtime
if nargin < 8 || isempty(varargin{6})
    verbose = 'on'; else verbose = varargin{6}; end 

% -- step 0
x = x0;
flag = 1;

% every n_adjust_residual iterations the residual is computed using the 
% definition 
n_adjust_residual = 30;
% n_adjust_residual = Inf; % 20130901

% Relative tolerance
r = b - A(x); % is the gradient of F
normr = norm(r(:));  % Norm of residual

switch stopping_rule
    case 1       % convergence if ||r(i)|| < tol*||b||
        n2b = norm(b(:));
        tolr = tol * n2b;
    case 2       % convergence if ||r(i)|| < tol*(||A||*||x(i)||+||b||)
        nfA = estimate_trace(A,size(b),'fro_norm');
        n2b = norm(b(:));
        n2x = norm(x(:));
        tolr = tol*(nfA*n2x+n2b);
    case 3       % convergence if ||r(i)|| < tol*||r0||
        normr0 = normr;
        tolr = tol * normr0;
    otherwise
        error('not implemented stopping criterion');
end


if (normr <= tolr)   % Initial guess is a good enough solution
   flag = 0;
   iter = 0;
   resvec = normr;
   func=0;%20130902
   return
end

resvec = zeros(maxit+1,1);         
resvec(1) = normr;                 
if nargout == 5, 
    func = zeros(maxit+1,1);
    func(1) = evaluate_func(A,b,x);
end 
                  
rho = 1;
% loop over maxit iterations (unless convergence or failure)

if isempty(M)
    %%% loop for conjugate gradient
    for i = 1 : maxit   
       rho1 = rho;
       rho = norm(r(:),2)^2;   
       if ((rho == 0) || isinf(rho))
          flag = 4;
          disp(['failure diagnosis: residual at ' num2str(i) ...
              '-th iteration has null or infinite norm']);
          break
       end
       if (i == 1)
          p = r; % conjugate direction
       else
          beta = rho / rho1;
          if ((beta == 0) || isinf(beta))
             flag = 4;
             disp(['failure diagnosis: beta value at ' num2str(i) ...
              '-th iteration is null or infinite']);
             break
          end
          p = r + beta * p;
       end
       q = A(p);
       pq = real(sum(sum(conj(p).*q))); % - quadratic form p'Ap
       if ((pq <= 0) || isinf(pq))
          flag = 4;
          disp(['failure diagnosis: at ' num2str(i) ...
              '-th iteration quadratic form is negative of infinite']);
          break
       else
          alpha = rho / pq;
       end
       if isinf(alpha)
          flag = 4;
          disp(['failure diagnosis: alpha value at ' num2str(i) ...
              '-th iteration is null or infinite']);
          break
       end
    %    p = p-mean(p(:));
       x = x + alpha * p;               % form new iterate

       %%%
    %    imagesc(log(abs(hilbert(x))));  
    %    colormap gray
    %    colorbar,
    %    drawnow,
       %%%

       if mod(i,n_adjust_residual) == 0
           r = b - A(x); % improves numerical stability
       else
           r = r - alpha * q;
       end
       normr = norm(r(:));
       if strcmp(verbose,'on')
           disp(['residual-norm at ' num2str(i) ...
               '-th iteration: ' num2str(normr)]);
       end
       resvec(i+1) = normr;
       if nargout == 5, func(i+1) = evaluate_func(A,b,x); end

       if stopping_rule == 1 || stopping_rule == 3
            if (normr <= tolr)               % check for convergence
                flag = 0;
                iter = i;
                break
            end
       end

       if stopping_rule == 2
           n2x = norm(x(:));
           tolr = tol*(nfA*n2x+n2b);
           if (normr <= tolr)               % check for convergence
                flag = 0;
                iter = i;
                break
            end
       end

    end % for i = 1 : maxit

    else % if isempty(M)
    %%% loop for preconditioned conjugate gradient

    for i = 1 : maxit   
       rho1 = rho;
       z = M(r); % preconditioned residual
       rho = real(sum(sum(conj(r).*z)));   
       if ((rho == 0) || isinf(rho))
          flag = 4;
          disp(['failure diagnosis: preconditioned residual at ' num2str(i) ...
              '-th iteration has null or infinite norm. Preconditioner could be not positive definite']);
          break
       end
       if (i == 1)
          p = z; % conjugate direction
       else
          beta = rho / rho1;
          if ((beta == 0) || isinf(beta))
             flag = 4;
             disp(['failure diagnosis: beta value at ' num2str(i) ...
              '-th iteration is null or infinite']);
             break
          end
          p = z + beta * p;
       end
       q = A(p);
       pq = real(sum(sum(conj(p).*q))); % - quadratic form p'Ap
       if ((pq <= 0) || isinf(pq))
          flag = 4;
          disp(['failure diagnosis: at ' num2str(i) ...
              '-th iteration quadratic form is negative of infinite']);
          break
       else
          alpha = rho / pq;
       end
       if isinf(alpha)
          flag = 4;
          disp(['failure diagnosis: alpha value at ' num2str(i) ...
              '-th iteration is null or infinite']);
          break
       end
       x = x + alpha * p;               % form new iterate

       %%%
    %    imagesc(log(abs(hilbert(x))));  
    %    colormap gray
    %    colorbar,
    %    drawnow,
       %%%

       if mod(i,n_adjust_residual) == 0
           r = b - A(x); % improves numerical stability
       else
           r = r - alpha * q;
       end
       normr = norm(r(:));
       if strcmp(verbose,'on')
           disp(['residual-norm at ' num2str(i) '-th iteration: ' num2str(normr)]);
       end
       resvec(i+1) = normr;
       if nargout == 5, func(i+1) = evaluate_func(A,b,x); end
       if stopping_rule == 1 || stopping_rule == 3
    %        normr
            if (normr <= tolr)               % check for convergence
                flag = 0;
                iter = i;
                break
            end
       end

       if stopping_rule == 2
           n2x = norm(x(:));
           tolr = tol*(nfA*n2x+n2b);
           if (normr <= tolr)               % check for convergence
                flag = 0;
                iter = i;
                break
            end
       end

    end                                % for i = 1 : maxit
end % if isempty(M)
%%
% truncate the zeros from resvec
if flag == 1
    iter = maxit;
end

if flag == 0
   resvec = resvec(1:i+1);
   if nargout == 5;
       func = func(1:i+1);
   end
end
   

function F = evaluate_func(A,b,x)
% F = 1/2*x'Ax - b'x
% g = Ax-b (residual)

if isreal(x), a = .5; else a = 1; end
F = a*sum(sum(conj(x).*A(x))) - sum(sum(conj(x).*b));
F = real(F);
