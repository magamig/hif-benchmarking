function Y = func_blurringY(X,psf,varargin)
% Spatial bluring function
%  [dx dy] = size(psf);
%  [N_row N_col L] = size(X);
% if exist('trans','var')
%     Y = zeros([N_row*dx N_col*dy L]);
%     for l=1:L
%         Y(:,:,l)=kron(eye(N_row),ones(dx,1))*X(:,:,l)*kron(eye(N_col),ones(1,dy));% The size increases
%     end
%     Y=repmat(psf,[N_row N_col L]).*Y;    
% else
%     Y = zeros([N_row/dx N_col/dy L]);
%     %Step 1: The bluring process
%     X=repmat(psf,[N_row/dx N_col/dy L]).*X; % The size doesn't change
%     %Step 2: The decimation process
%     for l=1:L
%         Y(:,:,l)=kron(eye(N_row/dx),ones(1,dx))*X(:,:,l)*kron(eye(N_col/dy),ones(dy,1));
%     end
% end
    %Type 1: Convolution
%     Y = zeros(size(X));
%     if exist('trans','var')   
%         psf = psf';
%     end
%     for l=1:size(X,3)
%         Y(:,:,l) = imfilter(squeeze(X(:,:,l)),psf,'symmetric','same');
%     end

FBm  = fft2(psf.B);
FBmC = conj(FBm);
% build convolution filters
FZ = zeros(size(X));
FZC = zeros(size(X));
for i=1:size(X,3)
    FZ(:,:,i) = FBm;
    FZC(:,:,i) = FBmC;
end
ConvCBD = @(X,FK) real(ifft2(fft2(X).*FK)); 

% if size(X,3)==size(psf.dsp,3)
%     dsp_op=psf.dsp;
% elseif size(X,3)==size(psf.dsp_sub,3)
%     dsp_op=psf.dsp_sub;
% end
if nargin==2
    Y=ConvCBD(X,FZ);
%     Y = Y.*dsp_op;
    %    Y=Y(1:psf.ds_r:end,1:psf.ds_r:end,:);
elseif nargin==3
    dsp_op=repmat(psf.dsp,[1 1 size(X,3)]);
    X = X.*dsp_op;
    %    X=X(1:psf.ds_r:end,1:psf.ds_r:end,:);
    Y=ConvCBD(X,FZC);
end

    
    
    %Type 2: Bluring
%      tmp = kron(X(1:dx:end,1:dy:end,l),ones([dx dy]));
%      Y(:,:,l) = tmp(1:N_row,1:N_col);
     %Type 3: sum of small blocks
%      tmp1=toeplitz([1 zeros(1,N_row -1)],[ones(1,dx) zeros(1,N_row -dx)]);
%      tmp2=toeplitz([1 zeros(1,N_col -1)],[ones(1,dy) zeros(1,N_col -dy)])';
%      tmp1=kron(eye(N_row/dx),ones(1,dx));
%      tmp2=kron(eye(N_col/dy),ones(1,dy))';

%     tmp=kron(eye(N_row/dx),ones(1,dx))*X(:,:,l)*kron(eye(N_col/dy),ones(1,dy))';     
%     Y(:,:,l) = imresize(tmp,[N_row,N_col],'bilinear');
    