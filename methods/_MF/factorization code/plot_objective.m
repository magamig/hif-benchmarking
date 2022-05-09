function plot_objective(A1,X1,A2,X2,Y,numPts)

% Takes the line 
%
%    L = { t (X1,A1) + (1-t) (X2,A2) | t in [-1,1] }
%
% Projects it onto the solution manifold M, and tries to visualize the resulting 
%  objective ||X||_1
%
% The difficulty is the sign-permutation ambiguity: we assume that A1 is
% close enough to A2 that this can be resolved via a greedy algorithm. 
%
% Care should be taken when interpreting results...
%

SIGN_PERM = false;

% first handle the permutation ambiguity in a greedy manner

if SIGN_PERM,
    [P,Sigma] = greedy_sign_perm(A1,A2);

    disp(size(P));
    disp(size(A1));
    disp(size(A2));

    A2 = A2(:,P);
    X2 = X2(P,:);

    A1 = A1 * diag(Sigma);
    X1 = diag(Sigma) * X1; 
end
    
objVals = zeros(numPts,1);
tStep = 1 / (numPts - 1);
tVec = -1:tStep:1;

for i = 1:length(tVec),
    t = tVec(i);
    
    X = t * X1 + (1-t) * X2;
    A = t * A1 + (1-t) * A2;
    
    [A,X] = naive_projection(Y,A,X);
    %[A,X] = manifold_projection(Y,A,X);
        
    objVals(i) = sum(sum(abs(X))); 
end

figure(2); 
plot(objVals(end:-1:1),'b-','LineWidth',2);