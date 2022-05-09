function reconstructedHS = reconstruct_hyperspectral(dataRGB,dataHS_LR,P_rgb,A,X)

% reconstruct_hyperspectral
%
%   Inputs:
%     dataRGB   - 
%     dataHS_LR - 
%     A - 
%     X - 
%
%   Outputs:
%     reconstructedHS - 
%
%   Fall '10, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


K_TO_KEEP   =  6;           %%% How many basis elements should we retain?

ETA         = .5;           %%% Noise tolerance for basis pursuit denoise; ideally should be set according
                            %%%  to our idea of the noise level in the
                            %%%  data.                            

[w,h,s]    = size(dataHS_LR);
[W,H,dc]   = size(dataRGB);
blockW   =  W / w;
blockH   =  H / h;

reconstructedHS = zeros(W,H,s);

% MAIN ITERATION
for i = 1:h
    for j = 1:w

        X_block = X(:,i+w*(j-1));
        
        [X_sqr_sort,X_indices] = sort(abs(X_block),'descend');        
        kBlock = K_TO_KEEP;                  
        
        AI = [A(:,X_indices(1:kBlock))]; 
        PAI = P_rgb' * AI;       
        
        tStart = tic;
        for aa = 1:blockW,
            for bb = 1:blockH,                

                % solve the problem 
                %
                %    minimize    ||x||_1    
                %    subject to  ||q - A_I x || < eps
                %
                %    With eps chosen according to the noise level                    
                                                
                q = reshape(dataRGB( blockW*(i-1)+aa, blockH*(j-1)+bb, : ), 3, 1 );                                              
                x = bpdn_direct( PAI, q, 2^8 * ETA ); 
                reconstructedHS( blockW*(i-1)+aa, blockH*(j-1)+bb, : ) = AI * x;                                  
            end
        end
        
        tElapsed = toc(tStart);                
        disp(['Block (' num2str(i) ',' num2str(j) ') ... Nonzeros: ' num2str(kBlock) '   Computation time: ' num2str(tElapsed)]);        
        
    end
end
