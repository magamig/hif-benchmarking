function relative_error(A1,X1,A2,X2)

relativeErrorA = norm(A1-A2,'fro')/norm(A2,'fro');
relativeErrorX = norm(X1-X2,'fro')/norm(X2,'fro');

disp(relativeErrorX);
disp(relativeErrorA);