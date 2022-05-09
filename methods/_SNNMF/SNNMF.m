function SNNMF_OUT = SNNMF(filestem, roi, dsf, N, A, rescale, saveImages)
%SNNMF_EXP performs Sparse Non-Negative Matrix Factorization
tol = 1e-4; itercap = 5; % tolerance variables for global iterations
disp('* * * Loading Image Variables * * *');
dims = [0 0 -1 1;-1 1 0 0]*(roi') + [1;1]; %dims = [rows; cols] of the downsampled images
P = create_P(dsf,roi); %permutation matrix that makes PGG'P' block diagonal
F = create_F(); %spectral transform matrix
G = P*create_G(roi,dsf); %G is the spatial transform (downsampling) matrix; note we use P*G as G
Z_ground_truth = load_Z(filestem,roi,dsf)*(P');
X = create_X_SNNMF(Z_ground_truth, G, P, roi, dsf, rescale);
Y = create_Y(Z_ground_truth, F);
if (size(A,1) ~= size(X,1) || size(A,2) ~= N) %if A is not already initialized, initialize using AVMAX
    disp('* * * Initializing A via AVMAX * * *');
    A = AVMAX(X,N);
end
SNNMF_OUT.A = A; %record initial A
Lc = dims(1)*dsf*dims(2)*dsf; %Lc = number of high res pixels (size of each row of Z)
k = (dsf^2)/(1+dsf^2); %scaling parameter passed to ADMM_S that affects lambda when dsf is changed
t = (A')*(F')*F*A; [U D] = eig((t+t')/2);
disp('* * * Initializing S via ADMM * * *');
[S y] = ADMM_S(X,Y,F,G',A,zeros(N,Lc),zeros(N*Lc,1),U,D,dsf,k);
disp('* * * Saving First Reconstruction Variables * * *');
A_initial = A; S_initial = S; %save results at this stage
disp('* * * Updating A via ADMM * * *');
A = ADMM_A(S,G,F,X,Y,A);
t = (A')*(F')*F*A; [U D] = eig((t+t')/2);
disp('* * * Beginning Lambda-A Loop * * *');
E0 = get_error(A,S,X,Y,G,F); dE = 1e0; %global iteration error
iter = 1; str = [num2str(iter), ': dE ', num2str(dE), ' E ', num2str(E0)]; disp(str);
%Print out the current RMSE
rmse_old = fast_rmse(Z_ground_truth,A*S);
str = ['---------------------------RMSE: ',num2str(rmse_old),'    dRMSE: N/A']; disp(str);
%%%%%%%%%%
while abs(dE) > tol && iter < itercap %global loop, alternating ADMM for S and A
    disp('* * * Updating S via ADMM * * *');
    [S y] = ADMM_S(X,Y,F,G',A,S,y,U,D,dsf,k);
    disp('* * * Updating A via ADMM * * *');
    A = ADMM_A(S,G,F,X,Y,A);
    t = (A')*(F')*F*A; [U D] = eig((t+t')/2);
    % Update stopping variables
    Ef = get_error(A,S,X,Y,G,F); dE = (E0-Ef)/E0; E0 = Ef; %global iteration error
    iter = iter + 1; str = ['Outer Iteration ',num2str(iter), ': dE ', num2str(dE), ' E ', num2str(E0)]; disp(str);
    %Print out the current RMSE
    rmse = fast_rmse(Z_ground_truth,A*S);
    str = ['---------------------------RMSE: ',num2str(rmse),'    dRMSE: ',num2str((rmse_old-rmse)/rmse_old)]; disp(str);
    rmse_old = rmse;
    %%%%%%%%%%
end
disp('* * * Saving Second Reconstruction Variables* * *');
A_iterated = A; S_iterated = S; %save final reconstruction variables
disp('* * * Alternating Back-Projection * * *');
Z = alternating_back_projection(A*S,X,Y,F,G);
disp('* * * Reconstructing Images * * *');
SNNMF_OUT.INITIAL = reconstruct_images(A_initial*S_initial*P,Y*P,dims,dsf,roi,filestem,saveImages);
SNNMF_OUT.ITERATED = reconstruct_images(A_iterated*S_iterated*P,Y*P,dims,dsf,roi,filestem,saveImages);
SNNMF_OUT.BACKPROJECTED = reconstruct_images(Z*P,Y*P,dims,dsf,roi,filestem,saveImages);
disp('* * * Finished * * *');
end

