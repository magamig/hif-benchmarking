% Function calculating the value of the response surface for different mu
function [J_1, J_2, J_3, X_fin]=calculate_J(mu, eta, par, R, X_CNN, HSI3, MSI3, sf)
    [nr,nc,~] = size(X_CNN);
    HR_HSI3=hyperConvert2D(X_CNN);
    H1=eta * (R'*R) + mu*eye(size(R,2));
    HHH1=par.HT(HSI3);
    H3=eta * (R'*MSI3)+mu*HR_HSI3+HHH1;
    X_fin=Sylvester(H1,par.fft_B,sf,nr/sf,nc/sf,H3); %% Sylvester equation (5)
    J_1 = norm(HSI3 - par.H(X_fin), 'fro').^2;
    J_2 = norm(MSI3 - R*X_fin, 'fro').^2;
    J_3 = norm(X_fin - HR_HSI3, 'fro').^2;
end