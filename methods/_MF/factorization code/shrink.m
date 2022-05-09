function Ms = shrink(M,lambda)

Ms = sign(M) .* pos( abs(M) - lambda );