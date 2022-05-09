function [X]=PMinus(X,n_dr,n_dc)
Y=X;Y(1:n_dr,1:n_dc,:)=0;%[nr,nc,nb]=size(X);
Y=PPlus(Y,n_dr,n_dc);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Sum according to the column
% Temp=reshape(Y,[nr*n_dc nc/n_dc nb]);%(:,:,i)
% Temp(:,1,:)=sum(Temp,2);
% % Temp1=reshape(Temp(:,1,:),[nr n_dc nb]); 
% %% Sum according to the row
% Temp1=reshape(permute(reshape(Temp(:,1,:),[nr n_dc nb]),[2 1 3]),[n_dr*n_dc nr/n_dr nb]); % Only process the low-frequency information
% Y(1:n_dr,1:n_dc,:)=permute(reshape(sum(Temp1,2),[n_dc n_dr nb]),[2 1 3]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X(1:n_dr,1:n_dc,:)=X(1:n_dr,1:n_dc,:)-Y(1:n_dr,1:n_dc,:);
%% Abadoned implementation
% [nr,nc,nb]=size(X);
% for i=1:nb
%     %% X: matrix of size nr*nc
%     X1=X(1:n_dr,1:n_dc,i);
%     X(1:n_dr,1:n_dc,i)=0;
%     %% Minus according to the column
%     Temp=im2col(X(:,:,i),[nr n_dc],'distinct');
%     Temp(:,1)=Temp(:,1)+sum(Temp(:,2:end),2);
%     Temp=col2im(Temp,[nr n_dc],[nr nc],'distinct');
%     %% Minus according to the row
%     Temp=im2col(Temp,[n_dr nc],'distinct');
%     Temp(:,1)=Temp(:,1)+sum(Temp(:,2:end),2);
%     Y_temp=col2im(Temp,[n_dr nc],[nr nc],'distinct');
%     Y(1:n_dr,1:n_dc,i)=X1-Y_temp(1:n_dr,1:n_dc);
% end
%% Prove that all the operation is invertiable
%     % Test
%     tempX=fft2(reshape((Q\VX_real)',[nr nc nb_sub])).*FZ_s;
%     for i=1:nb_sub
%        tempX(:,:,i)=PPlus(tempX(:,:,i),nr,nc,n_dr,n_dc);       
%     end
%     for i=1:nb_sub
%         X_est(:,:,i)=ifft2(PMinus(tempX(:,:,i),nr,nc,n_dr,n_dc)./FBm);
%     end
%     VX_est=real(Q*conv2mat(X_est,nb_sub));