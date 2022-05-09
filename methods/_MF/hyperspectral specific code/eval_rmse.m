function r = eval_rmse( A, B, scale )

res = A - B;
r = scale * sqrt( sum(sum(sum(res .* res ))) / prod(size(res)) ); 