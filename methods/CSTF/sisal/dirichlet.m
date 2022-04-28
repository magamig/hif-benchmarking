function [r] = dirichlet(a,dim)

%DIRICHLET Random matrices from dirichlet distribution.
%   R = DIRICHLET_RND(A,DIM) returns a matrix of random numbers chosen   
%   from the dirichlet distribution with parameters vector A.
%   Size of R is (N x N) where N is the size of A or (N x DIM) if DIM is given.
%
%
% Modification of the dirichlet_rnd function:
% A standard Dirichlet distribution is obtained from independent gamma
% distribuitions with  scale parameters a1,...,ak, and shape set to 1
%
% Author: Jose Bioucas Dias, 2004/10.


if nargin < 1, error('Requires at least one input arguments.'); end
if nargin > 2, error('Requires at most two input arguments.'); end

[rows columns] = size(a); 
if nargin == 1, dim = rows * columns; end
if nargin == 2, 
   if prod(size(dim)) ~= 1, error('The second parameter must be a scalar.'); end
end
if rows~=1 & columns~=1, error('Requires a vector as an argument.'); end

% fastest method that generalize method used to sample from
% the BETA distribuition: Draw x1,...,xk from independent gamma 
% distribuitions with  scale and  parameters a1,...,ak, and shape set to
% one, for each j let rj=xj/(x1+...+xk).


N = rows * columns;
for i = 1 : N
    % generates dim random variables 
    x(:,i) = gamrnd(a(i),1,dim,1); % generates dim random variables 
end                                   % with gamma distribution
r = x./repmat(sum(x,2),[1 N]);

% For more details see "Bayesian Data Analysis" Appendix A




