function [reconstructedHS,coefficientMap] = reconstruct_hyperspectral(dataRGB,dataHS_LR,P_rgb,A,X,groundTruthHS)

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


USE_BACKPROJECTION = false; %%% Enforce that the solution exactly argees with the observation via 
                            %%%  minimum L2 norm backprojection
CLIP      = false;          %%% Enforce that the solution lies in [0,2^16] by truncation
VISUALIZE = true;           %%% Make pictures? 
ETA       = 10/256;          %%% Noise tolerance for basis pursuit denoise; ideally should be set according
                            %%%  to our idea of the noise level in the
                            %%%  data.                            

[wLow,hLow,sLow] = size(dataHS_LR);
[w,h,dontCare]   = size(dataRGB);

displayRGB = zeros(w,h,3); 
numSpectra = size(P_rgb,1);

blockW = w / wLow;
blockH = h / hLow;

reconstructedHS = zeros(w,h,sLow);

nnzMap   = zeros(w,h); 
coefficientMap = zeros(w,h,size(A,2)); 

kMap = zeros(wLow,hLow);      % just for display - number of nonzeros needed to well-represent each region
normq = sqrt(sum( dataRGB .* dataRGB, 3 )); 


% MAIN ITERATION
for i = 1:wLow
    for j = 1:hLow  

        X_block = X(:,i+wLow*(j-1));
        
        [X_sqr_sort,X_indices] = sort(abs(X_block),'descend');   
        
        [X_sort,X_indices] = sort(abs(X_block),'descend');
        T_sort = cumsum(X_sort) / sum(X_sort);
        kBlock = min(find(T_sort > .90));
        
        %kBlock = 10;
        
        kMap(i,j) = kBlock;
                        
        AI = [A(:,X_indices(1:kBlock))]; 
        P_AI = P_rgb' * AI;       
                       
        RGB_pseudoInverse = pinv(P_rgb'); 
                
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
                
                H  = AI; 
                PH = P_AI; 
                                               
                x = bpdn_direct( PH, q, ETA ); 
                reconstructedHS( blockW*(i-1)+aa, blockH*(j-1)+bb, : ) = H * x;         
                
               
                if USE_BACKPROJECTION, 
                    reconstructedHS( blockW*(i-1)+aa, blockH*(j-1)+bb, : ) = reconstructedHS( blockW*(i-1)+aa, blockH*(j-1)+bb, : ) + reshape( RGB_pseudoInverse * (q - P_rgb' * AI * x), 1, 1, numSpectra ); 
                end
                
                if CLIP, 
                    reconstructedHS( blockW*(i-1)+aa, blockH*(j-1)+bb, : ) = min( max( reconstructedHS( blockW*(i-1)+aa, blockH*(j-1)+bb, : ), 0 ), 2^16 );
                end
                
                % for visualization
                nnzMap(blockW*(i-1)+aa, blockH*(j-1)+bb) = sum( abs( x ) > 1e-8 );                 
                coefficientMap( blockW*(i-1)+aa, blockH*(j-1)+bb, X_indices(1:kBlock)+1 ) = reshape(x,1,1,length(x));
              
            end
        end
        
        tElapsed = toc(tStart);
        
        curBlockRGB = dataRGB(  blockW*(i-1)+1:blockW*i, blockH*(j-1)+1:blockH*j, :  );           
        displayRGB(  blockW*(i-1)+1:blockW*i, blockH*(j-1)+1:blockH*j, :  ) = curBlockRGB;
                            
        disp(['Block (' num2str(i) ',' num2str(j) ') ... Nonzeros: ' num2str(kBlock) '   Signal level: ' num2str(2^-8 * sqrt(sum(sum(sum(curBlockRGB .* curBlockRGB)))/ prod(size(curBlockRGB)))) '   Computation time: ' num2str(tElapsed)]);        
    end
end
