function xOpt = bpdn_direct(A,y,eta)

[m,n] = size(A);
xOpt = zeros(n,1);

% special case: check if the zero solution works
if norm(y) < eta, return; end;

Aty = A'*y; 
GramMtx = A' * A; 

% if not, try every possible support size from 1 ... m 
for k = m:-1:1, 

    allI = nchoosek(1:n,k);
    P = randperm(size(allI,1));
    allI = allI(P,:);
    D = (0:2^k-1)';
    allSigma = 2 * transpose(rem(floor(D*pow2(-(k-1):0)),2)) - 1;
    P = randperm(size(allSigma,2));
    allSigma = allSigma(:,P);
    
    for i = 1:size(allI,1),

        I  = allI(i,:);
        v  = 1:n; 
        v(I) = 0; 
        Ic = (v ~= 0); 
        
        A_I  = A(:,I);
        A_Ic = A(:,Ic); 

        G = inv(A_I'*A_I);    
        
        p = y - A_I * G * (A_I' * y);

        c = p'*p - eta*eta;

        for j = 1:size(allSigma,2),
            sigma = allSigma(:,j);

            G_sigma = G * sigma;        
            q = 2 * A_I * G_sigma; 

            b = 2*(p'*q);
            a = q'*q;

            disc = b*b - 4 * a * c; 

            tRealPos = [];
            
            if disc >= 0, 
                sqrtDisc = sqrt(disc);

                r2 = ( -b + sqrtDisc ) / (2*a);
                r1 = ( -b - sqrtDisc ) / (2*a);

                if r1 > 0,
                    tRealPos = [r1 r2];
                elseif r2 > 0, 
                    tRealPos = [r2];
                end
            end

            for l = 1:length(tRealPos),
                
                beta = tRealPos(l);                
                xl = G*(A_I'*y) - (2*beta)*G_sigma;                
                h  = (2 * beta)^-1 * ( Aty - GramMtx(:,I) * xl );
                
                xTest = zeros(n,1);
                xTest(I) = xl;     
                
                maxNorm = max(abs(h(Ic)));
                                              
                if (isempty(maxNorm) || maxNorm <= 1+1e-10) && (max(abs(h(I)-sign(xl))) < 1e-6), 
                    optFound = true;
                    xOpt(I) = xl;
                    return; 
                end
            end
        end
    end
end