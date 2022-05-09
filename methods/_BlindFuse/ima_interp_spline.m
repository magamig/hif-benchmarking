function X_interp=ima_interp_spline(X,ds_r)
% Example for the usage of interp2
% Bicubic smoothing of an image
X = padarray(X,[1 1],'symmetric');
nr = size(X,1);nc = size(X,2);
% [x,y]   = meshgrid(1:nr,1:nc);
% [xi,yi] = meshgrid(1:1/ds_r:nr+1-1/ds_r,1:1/ds_r:nc+1-1/ds_r);
X_interp=zeros(nr*ds_r,nc*ds_r,size(X,3));
for k = 1:size(X,3)
% 	X_interp(:,:,k) = interp2(x,y,X(:,:,k),xi,yi,'spline');
    X_interp(:,:,k) = interp2(1:nc,1:nr,X(:,:,k),[1:1/ds_r:nc+1-1/ds_r]',1:1/ds_r:nr+1-1/ds_r,'spline');%spline nearest cubic linear
% Ref: http://bbs.06climate.com/forum.php?mod=viewthread&tid=11294
end
X_interp=X_interp(ds_r+1:end-ds_r,ds_r+1:end-ds_r,:);
% figure;
% subplot(1,2,1);imagesc(mean(X,3));% axis square;
% subplot(1,2,2);imagesc(mean(X_interp,3));% axis square;