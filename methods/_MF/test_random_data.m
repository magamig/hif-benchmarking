% test_random_data
%
%   Generates a random square problem and solves it

addpath('helpers');
addpath('factorization code');

m = 50;
k = 6;
n = m;
p = 1000;

% generate a random problem
I = columnwise_uniform_support_pattern(n,p,k);
A_true = normalize_columns( randn(m,n) );
X_true = I .* randn(n,p);
Y = A_true * X_true;

A0 = normalize_columns(transpose(randn(n,m)));               
X0 = pinv(A0) * Y;

[A_hat,X_hat] = dl_iterative_L1(Y,A0,X0);

% mod out sign-permutation ambiguity
[P,sigma] = greedy_sign_perm(A_hat,A_true);
A_hat = A_hat * diag(sigma);
X_hat = diag(sigma) * X_hat;
A_true = A_true(:,P);
X_true = X_true(P,:);

relativeErrorA = norm(A_hat-A_true,'fro')/norm(A_true,'fro');
relativeErrorX = norm(X_hat-X_true,'fro')/norm(X_true,'fro');

disp(' ');
disp('FACTORIZATION COMPLETE');
disp(' ');
disp('   Relative error in X:');
disp(relativeErrorX);
disp('   Relative error in A:');
disp(relativeErrorA);