function z = soft_neg(y,tau)
%  z = soft_neg(y,tau);
%
%  negative soft (proximal operator of the hinge function)

z = max(abs(y+tau/2) - tau/2, 0);
z = z./(z+tau/2) .* (y+tau/2);