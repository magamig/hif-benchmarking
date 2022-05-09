%% Post processing
% remove a thin frame in the image boundaries
% function []=PostProces(Zim,Xmh,Xp,Z,p,nc,nr,nb)
% frame_size=0;
% Zim = Zim(1+frame_size:end-frame_size,1+frame_size:end-frame_size,: );
% X_real = X_real(1+frame_size:end-frame_size,1+frame_size:end-frame_size,: );
% Xp= Xp(1+frame_size:end-frame_size,1+frame_size:end-frame_size,:);

figure(2)
imagesc(Zim(:,:,1))
title('estimated band   no 1')

% show erros in pan channel
% errHS = Zim - X_real;
% figure(3); figure(gcf);
% imagesc(mean(errHS,3));
% colorbar;
% title('2-norm error of HS image')

% show erros in ms  channel
% for i=1:N_band
%     errMs(:,:,i) = Zim(:,:,i)- X_real(:,:,i);
%     figure(300)
%     %mean_x = max(max(Xmo(:,:,i)));
%     imagesc(errMs(:,:,i))
%     colorbar
% end

% T0tal error
err = sqrt(sum((VX_est(1:N_band,:)- conv2mat(X_real,N_band)).^2));
figure(4)
imagesc(conv2im(err,1))
colorbar;
title('2-norm error of target image')

% plot spectra
Xmhmat = conv2mat(X_real,N_band);
figure(5);
% index = sub2ind([nc,nr],[13 13 13 13],[55,56,80,81]);
index = sub2ind([nc,nr],[13 13 13 13],[5,8,12,15]);
plot(VX_est(1:N_band,index),'b','LineWidth',2)
hold on
plot(Xmhmat(:,index),'r','LineWidth',2)
hold off
legend('Estimated', 'Original')
title('MS spectra')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('proposed method \n')
%Delete the 6th band because it is nearly zero-mean
Zim(:,:,6)=[];

[err_max,err_l1,err_l2,Q,SAM_hat,RMSE_hat,ERGAS,D_dist] = metrics(X_real_sub,Zim,psfY.ds_r);
err_max_mean = mean(err_max); %.*cm'
err_l1_mean  = mean(err_l1);  %.*cm'
err_l2_mean  = mean(err_l2);  %.*cm'
SNR_fusion   = 20*log10(norm(X_real_sub(:))/norm(X_real_sub(:)-Zim(:)));

SNR_set    =[SNR_set SNR_fusion]
RMSE_set   =[RMSE_set RMSE_hat]
mean_Q     =[mean_Q mean(Q)]
SAM_set    =[SAM_set SAM_hat]
ERGAS_set  =[ERGAS_set ERGAS]
D_dist_set =[D_dist_set D_dist]

% metrics = imagemetrics(X_real, Xp,Zim(:,:,1:p-1));
% fprintf('\n ERGAS = %2.2f\n CAVE = %2.2f\n RASE = %2.2f\n RMSE = %2.2f\n  SAM = %2.2f\n SID = %2.2f\n SPATIAL = %2.2f\n\n ', ...
%        metrics(2),metrics(3),metrics(4),metrics(5),metrics(6),metrics(7),metrics(8) )

% Correlarion coeficientX
% for i=1:N_band
%     a = Z(i,:);
%     b = reshape(X_real(:,:,i),1,nr*nc);
%     rho(i) = (a-mean(a))*(b-mean(b))'/ norm(a-mean(a))/norm(b-mean(b));
% end
% fprintf('correlation coefficient')
% rho;

% figure(6)
% semilogy(cost_fun(1:end))
% ylabel('cost function')
% 
% figure(10);plot(tau_d_set,RMSE_set,'o-','LineWidth',1);xlabel('${{\bf{\tau}}_d}/{\|{\bf N}_m\|_F^2}$','Interpreter','LaTex');ylabel('RMSE')



%%% PCA methd
% fprintf('PCA')
% Fpca=PCAfusion(conv2im(Ymlu,p-1),Xp);
% metrics = imagemetrics(X_real, Xp,Fpca);
% fprintf('\n ERGAS = %2.2f\n CAVE = %2.2f\n RASE = %2.2f\n RMSE = %2.2f\n  SAM = %2.2f\n SID = %2.2f\n SPATIAL = %2.2f\n\n ', ...
%        metrics(2),metrics(3),metrics(4),metrics(5),metrics(6),metrics(7),metrics(8) )