function [val] = get_error(A,S,X,Y,G,F)
% Get the Alternating Lambda-A Update Error
val = norm(X-A*S*G,'fro')^2 + norm(Y-F*A*S,'fro')^2;
end