function x_hat = constrained_shrinkage(y,z,w,beta,eta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Solves the optimization problem
%
%      minimize     ||x||_{w1} + beta ||x-y||^2
%      subject to   ||x-z|| <= eta
%
%   where ||x||_{w1} is the w-weighted L1 norm
%
%      ||x||_{w1} = \sum_i w_i |x_i|
%
%   Inputs:
%      y - m x 1 vector
%      z - m x 1 vector
%      w - m x 1 weight vector
%      beta - weight on the L2 norm
%      eta  - clamp on distance to z
%
%   Outputs:
%      x_hat - solution
%
%   Uses an O(m log m)-time algorithm, may be slow in iterpreted code (?)
%
%   Spring 2010, John Wright. Questions? jnwright@uiuc.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE = 0;

x_hat = shrink( beta * y, w/2 ) / beta;
if norm(x_hat - z) < eta,
	return;
end
    
xi_minus_vec = beta * ( y - z ) + sign(z) .* w / 2;
xi_plus_vec  = beta * ( y - z ) - sign(z) .* w / 2;

tau_minus_vec = - beta * y ./ z - w ./ (2 * abs(z));
tau_plus_vec  = - beta * y ./ z + w ./ (2 * abs(z));

tau_vec             = [ tau_minus_vec; tau_plus_vec ];
num_increment_vec   = [ - xi_minus_vec .* xi_minus_vec; xi_plus_vec .* xi_plus_vec ];
denom_increment_vec = [ - z .* z;  z .* z ];

[tau_sort,ind] = sort(tau_vec,'ascend');

tau_vec = [tau_sort; inf];
num_increment_vec   = num_increment_vec(ind);
denom_increment_vec = denom_increment_vec(ind);

num_vec   = sum(xi_minus_vec .* xi_minus_vec) + cumsum(num_increment_vec);
denom_vec = eta*eta + cumsum(denom_increment_vec);

tau_opt_vec = sqrt( num_vec ./ denom_vec ) - beta;

iOpt = min( find( (tau_opt_vec < tau_vec(2:end)) .* (tau_opt_vec > 0) ) );

tau_hat = tau_opt_vec(iOpt);

%if isempty(tau_hat),
%    disp('TAU HAT IS EMPTY!');
%    disp(sum(isnan(tau_opt_vec)));
%end

if isempty(tau_hat),
    x_hat = z;
else
    x_hat = shrink( beta * y + tau_hat * z, w/2 ) / (beta + tau_hat);
end