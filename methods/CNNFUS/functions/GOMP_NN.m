function [ aa ] = GOMP_NN( D,Y,L,gamma)
% [m, n_p] = size(Y);
[m, k] = size(D);
Dn = normc(D);
R = Y;
S = [];
    while 1
        R0 = R;
        [v, indx] = sort(Dn' * normc(R), 'descend');
        j = indx(1:L,1);
        S = union(S, j');   
        
        aa = zeros(k,1);
            a = lsqnonneg(D(:,S), Y);
            aa(S,1) = a;
%             Aa = [Aa aa];
            R = Y- D(:,S)*a;
    
        if norm(R,2) > gamma*norm(R0,2)      
            break;
        end
    end
end

