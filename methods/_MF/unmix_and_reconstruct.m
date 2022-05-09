function [recHS,facBasis,facCoeffs] = unmix_and_reconstruct( dataHS, dataRGB, P_rgb )

% unmix_and_reconstruct
%
%   Inputs: 
%      dataHS  - low-resolution hyperspectral observation,  w x h x s
%      dataRGB - high-resolution RGB observation, W x H x 3
%      P_rgb   - s x 3, columns are the RGB filter responses
%
%   Outputs:
%      recHS     - high-resolution hyperspectral estimate, W x H x s
%      facBasis  - s x s, sparsifying basis for the low-resolution spectra
%                         sorted in descending order of L1 contribution
%      facCoeffs - w x h x s, the coefficients of each spectral observation
%                         with respect to the basis, arranged in image form
%
%  

[w,h,s]  = size(dataHS);
[W,H,dc] = size(dataRGB);
Y = zeros(s,w*h);
for i = 1:w,
    for j = 1:h,
        Y(:,i+h*(j-1)) = dataHS(i,j,:);
    end
end

A0 = normalize_columns(randn(s,s));
X0 = pinv(A0) * Y;

%
disp('   Performing matrix factorization');

[A,X] = dl_iterative_L1(Y,A0,X0);

facCoeffs = zeros(w,h,s);
for i = 1:s,
    facCoeffs(:,:,i) = reshape(X(i,:),w,h);
end

[dc,I] = sort(sum(abs(X),2),'descend');
facBasis  = A(:,I);
facCoeffs = facCoeffs(:,:,I);

disp('   Reconstructing the hyperspectral image');
recHS = reconstruct_hyperspectral(dataRGB,dataHS,P_rgb,A,X);

