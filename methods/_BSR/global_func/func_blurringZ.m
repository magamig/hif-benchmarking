function Z = func_blurringZ(X,psf)
X_vec=reshape(X,[size(X,1)*size(X,2) size(X,3)]);
Z=psf*X_vec';
Z=reshape(Z',[size(X,1) size(X,2) size(psf,1)]);

% FBp  = fft2(psf);
% FBpC = conj(FBp);
% % build convolution filters
% L=1; % Panchromatic Image
% FZ = zeros(nr,nc,L);
% FZC = zeros(nr,nc,L);
% for i=1:size(X,L)
%     FZ(:,:,i)  = FBp;
%     FZC(:,:,i) = FBpC;
% end
% ConvCBD = @(X,FK)  real(ifft2(fft2(X).*FK));
% if exist('trans','var')
%     Z=ConvCBD(X,FZ);
% else
%     Z=ConvCBD(X,FZC);
% end