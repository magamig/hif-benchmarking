% Load the real image % X = imread('peppers.png');X = double(X(1:2:end,1:2:end,:));X = X/max(X(:));
function [X_real,band_remove]= real_image(Input,nr,nc,N_band,N_start_r,N_start_c)
Nset_r=N_start_r+1:N_start_r+nr;
Nset_c=N_start_c+1:N_start_c+nc;
if strcmp(Input,'moffet_ROI3_bis.mat')
    load(Input);%load('moffet_ROI3_bis.mat');
    % Remove the water absorbed bands
    %     im=im(:,:,mean(mean(im,2),1)~=0);
    %     im=im(:,:,max(max(im,[],2),[],1)<10000);
    band_remove=[1:6 12 103:117 150:170 220:224];
    im(:,:,band_remove)=[];
    im=(im-5*min(im(:)))/500;
%     im=(im-1000)/500; 
    im= im/max(im(:));
    Start_band=1;band=size(im,3);im=im(Nset_r,Nset_c,Start_band:Start_band+band-1);
    if N_band == 3
        X(:,:,1)=mean(im(:,:,1:band/3),3);          %X(:,:,1)=X(:,:,1)-min(min(X(:,:,1)));X(:,:,1)=X(:,:,1)/max(max(X(:,:,1)));
        X(:,:,2)=mean(im(:,:,band/3+1:band/3*2),3); %X(:,:,2)=X(:,:,2)-min(min(X(:,:,2)));X(:,:,2)=X(:,:,2)/max(max(X(:,:,2)));
        X(:,:,3)=mean(im(:,:,band/3*2+1:band),3);   %X(:,:,3)=X(:,:,3)-min(min(X(:,:,3)));X(:,:,3)=X(:,:,3)/max(max(X(:,:,3)));
    else
        X=im(:,:,1:1:N_band);
    end
%% Gaussian data
%     im=randn(200,200,200);
%     X=randn([128 128 3])*0.1+0.8;
%% non-Gaussian data
%     X(1:64,:,:)  =randn([64 128 3])*0.1+0.3;
%     X(65:128,:,:)=randn([64 128 3])*0.1+0.7;
elseif strcmp(Input,'pavia.mat')
    X = double(imread('original_rosis.tif'));band_remove=[];
    X=X(:,1:end-1,:); %remove last column
    X=X(:,:,11:end);  %remove the first 11 bands        
    X=X(Nset_r,Nset_c,1:N_band);
    X=X+mean(X(:)); % This is adding an offset to make sure there is no negative data
    X=X/(max(X(:))+min(X(:)));
elseif strcmp(Input,'pleiades_subset.mat')
    load pleiades_subset.mat;
    X=Xmh(Nset_c,Nset_c,1:N_band);
%     X=double(imread('peppers.png'));
%     X=X(1:N_row,1:N_col,1:N_band);
else
    X(1:32,:,:)=randn([32 64 3])*0.1+0.3;
    X(33:64,:,:)=randn([32 64 3])*0.1+0.7;
end
% for i=1:size(X,3)
%     Im = X(:,:,i);%-mean2(X(:,:,i));
%     X(:,:,i) = Im/sqrt(mean(Im(:).^2));
% end
% X = X/max(X(:));

% X = imadjust(X,stretchlim(X),[]);%R_real=corrcoef(reshape(X_real,[N_row*N_col L]));
X_real = X; 
% miu_x_real=squeeze(mean(mean(X_real,1),2)); s2_real = cov(reshape(X_real,[size(X,1)*size(X,2) size(X,3)]));

% figure(111); for i=1:size(X,3); temp=X_real(:,:,i);subplot(size(X,3),1,i); hist(temp(:),1000); end
% figure(222); for i=1:size(X,3); subplot(1,size(X,3),i); imagesc(X_real(:,:,i)); colormap gray; end