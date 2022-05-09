function [VXM,VXH,FBm,FBmC,FZ,FZ_s,FZC,FZC_s,mask_s,ConvCBD,conv2im,conv2mat] = func_define(XM,XH,psfY,nb_sub)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Define functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define a circular convolution (the same for all bands) accepting a matrix and returnig a matrix 
[nr nc ~]=size(XM);N_band=size(XH,3);
ConvC    = @(X,FK,nb) reshape(real(ifft2(fft2(reshape(X', nr,nc,nb)).*repmat(FK,[1,1,nb]))), nr*nc,nb)';
% define a circular convolution (band dependent) accepts a matrix and returns a matrix 
ConvCBD  = @(X,FK)    reshape(real(ifft2(fft2(reshape(X', nr,nc,size(X,1))).*FK)), nr*nc,size(X,1))';
% convert matrix to image
conv2im  = @(X,nb)    reshape(X',nr,nc,nb);
% convert image to matrix
conv2mat = @(X,nb)    reshape(X,nr*nc,nb)';
% ms SNR per band  in dBs
mask=psfY.dsp;  
%--------------------------------------------------------------------------
%   build matrices versions of the images
%--------------------------------------------------------------------------
VXM    = conv2mat(XM,size(XM,3));
mask   = repmat(conv2mat(mask,1),N_band,1); % replicate mask to size(Z)
mask_s = mask(1:nb_sub,:);

VXH  = conv2mat(XH,size(XH,3));
VXHu = VXH;  % save for other methods
VXH  = VXH.*mask;
%-----------------------------------------------------------------------
% degrade convolution kernel to test the robustnes of the methods
%------------------------------------------------------------------------
% define convolution filter
Bm = psfY.B; % model the MR blur
%% build convolution filters
FBm   = fft2(Bm);                    FBmC  = conj(FBm);
FZ    = repmat(FBm,[1 1 N_band]);    FZ_s  = FZ(:,:,1:nb_sub);
FZC   = repmat(FBmC,[1 1 N_band]);   FZC_s = FZC(:,:,1:nb_sub); %should be FZC instead of FZ. Whole
% II_FB = repmat(1./(abs(FBm.^2) + 2),[1 1 N_band]);