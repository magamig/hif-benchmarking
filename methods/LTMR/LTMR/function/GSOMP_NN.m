function [ Aa ] = GSOMP_NN( D,Y,L,gamma)
[m, n_p] = size(Y);
[m, k] = size(D);
Dn = normc(D);
R = Y;
S = [];
    while 1
        R0 = R;
        [v, indx] = sort(sum((Dn' * normc(R)),2), 'descend');
        j = indx(1:L,1);
        S = union(S, j');   
        Aa = [];
        aa = zeros(k,1);
        for i =1:n_p        
            a = lsqnonneg(D(:,S), Y(:,i));
            aa(S,1) = a;
            Aa = [Aa aa];
            R(:,i) = Y(:,i)- D(:,S)*a;
        end   
        if norm(R,2) > gamma*norm(R0,2)      
            break;
        end
    end
end


