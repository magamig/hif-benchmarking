% Function searching the optimization using golden section
function [X_fin, mu_opti, ita]=search_2_gss(par, R, X_CNN, HSI3, MSI3, sf)
 
    a = 10^(-8);
    b = 1;
    ell = 0.001;

    alpha = (3 / 31)^2;
    beta = (1 / sf^2)^2;
    ita = 1;
    
    gr = (sqrt(5) + 1) / 2;   
    c = b - (b - a) / gr; 
    d = a + (b - a) / gr; 

%     iter = 0;
    while abs(c - d) > ell
        dis_1 = MDC_dis(c, ita, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta);
        dis_2 = MDC_dis(d, ita, par, R, X_CNN, HSI3, MSI3, sf, alpha, beta);
        if dis_1 < dis_2
            b = d;
        else
            a = c;
        end
        c = b - (b - a) / gr; 
        d = a + (b - a) / gr; 
%         iter = iter + 1;
    end
%         disp(iter);   
    mu_opti = ((b+a)/2) * (alpha + beta);
    [~, ~, ~, X_fin] = calculate_J(mu_opti, ita, par, R, X_CNN, HSI3, MSI3, sf);

end