 function  ni = find_n(X, k)
% X=normc(X);
			D = L2_distance(X, X);
            [foo, ind] = sort(D, 1);
          ni=ind(1:k,:);