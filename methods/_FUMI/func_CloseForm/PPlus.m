function [X]=PPlus(X,n_dr,n_dc)
%% Input X: matrix of size nr*nc
%% Each block of X: matrix of size n_dr*n_dc
%% This function adds all the blocks to the 1st block, i.e., multiply by P^{-1}
% e.g. The input is 
%      X=[X11 X12 X13 X14;
%         X21 X22 X23 X24;
%         X31 X32 X33 X34;
%         X41 X42 X43 X44]
% The output is 
%      X=[\sum_{i,j}X_{i,j} X12 X13 X14;
%         X21 X22 X23 X24;
%         X31 X32 X33 X34;
%         X41 X42 X43 X44]
[nr,nc,nb]=size(X);
%% Sum according to the column
Temp=reshape(X,[nr*n_dc nc/n_dc nb]);
Temp(:,1,:)=sum(Temp,2);
% Temp1=reshape(Temp(:,1,:),[nr n_dc nb]); 
%% Sum according to the row
Temp1=reshape(permute(reshape(Temp(:,1,:),[nr n_dc nb]),[2 1 3]),[n_dc*n_dr nr/n_dr nb]); % Only process the low-frequency information
X(1:n_dr,1:n_dc,:)=permute(reshape(sum(Temp1,2),[n_dc n_dr nb]),[2 1 3]); %% Pay attention the shape of reshape: it is transposed
end
% tic
% X(1:n_dr,1:n_dc,:)=sum(cell2mat(reshape(mat2cell(X,n_dr*ones(1,nr/n_dr),n_dc*ones(1,nc/n_dc),nb),[1 1 1 (nr/n_dr)*(nc/n_dc)])),4);
% toc
%% Three implementation methods
%% Use im2col
% for i=1:nb
%     Temp=im2col(X(:,:,i),[nr n_dc],'distinct');
%     Temp=col2im(Temp,[nr n_dc],[nr nc],'distinct');
%     Temp=im2col(Temp,[n_dr nc],'distinct');
%     Y_temp=col2im(Temp,[n_dr nc],[nr nc],'distinct');
% end
%% reshape
%     Temp=reshape(Temp,[nr nc]);
%     Temp=reshape(Temp',[n_dr*nc nr/n_dr]);
%     Y_temp=reshape(Temp,[nc nr])';
%     Temp(:,1)=sum(Temp(:,1:end),2);%Temp(:,1)+
%     Y(1:n_dr,1:n_dc,i)=Y_temp(1:n_dr,1:n_dc);

% dr=nr/n_dr;dc=nc/n_dc;
% for i=1:nb;
%     X(1:n_dr,1:n_dc,i)=blockfun(X(:,:,i),[dr dc],@sum);
% end