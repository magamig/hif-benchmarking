% eval_soln.m        


[P,sigma] = greedy_sign_perm(A_hat,A_true);
A_hat = A_hat * diag(sigma);
X_hat = diag(sigma) * X_hat;
A_true = A_true(:,P);
X_true = X_true(P,:);

relativeErrorA = norm(A_hat-A_true,'fro')/norm(A_true,'fro');
relativeErrorX = norm(X_hat-X_true,'fro')/norm(X_true,'fro');

disp(relativeErrorX);
disp(relativeErrorA);