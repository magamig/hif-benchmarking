function Dis = MDC_dis(mu, ita, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta)
    [J_1, J_2, J_3, ~] = calculate_J(mu, ita, par, R, X_CNN, HSI3, MSI3, sf);
    J_3 = J_3 * (alpha + beta);   
    J_1 = J_1 + J_2;
    Dis = sqrt((J_1).^2 + (J_3).^2);
end