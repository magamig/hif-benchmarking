clear;warning off;
addpath([pwd,'\func_CloseForm']);
%% This script is to implement the joint fusion and unmixing of multi-band images
%If you use this code, please cite the following paper:

% [1] Q. Wei, J. M. Bioucas-Dias, N. Dobigeon, J-Y. Tourneret and S. Godsill,
% Blind Model-Based Fusion of Multi-band and Panchromatic Images,
% Proc. IEEE Int. Conf. Multisensor Fusion and Integr. for Intell. Syst. (MFI), 
% Baden-Baden, Germany, Sept. 2016, to appear.

%% Processing the MS+PAN data
normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),5),-5)/4+0.5;
%% Deimos-2 data loading
PAN=double(imread('DE2_PAN_L1C_000000_20150530T183126_20150530T183129_DE2_5119_FC14.tif'));const=max(PAN(:));
PAN=PAN/const;PAN=rot90(PAN,3);%PAN=permute(PAN,[2 1]);
MS=double(imread('DE2_MS4_L1C_000000_20150530T183126_20150530T183129_DE2_5119_FC14.tif'));MS=MS/const;MS=rot90(MS,3);%MS=permute(MS,[2 1 3]);
MSband_show=[2 3 4];
nr_start=300;nr_wid=500;
nc_start=300;nc_wid=500;

ratio=4;
MS=MS(nr_start:nr_start+nr_wid-1,nc_start:nc_start+nc_wid-1,:);
PAN=PAN((nr_start-1)*ratio+1:(nr_start-1+nr_wid)*ratio,(nc_start-1)*ratio+1:(nc_start-1+nc_wid)*ratio,:);

figure(1);imshow(normColor(MS(:,:,MSband_show)));
title('MS','FontSize',15);
figure(2);imshow(normColor(PAN)); 
title('PAN','FontSize',15);
drawnow;
%% Plot the observed MS and PAN images
zoom_1=[{1400:1500-1},{830:960-1}]; % Contest 1
zoom_2=[{1280:1380-1},{440:570-1}]; % Contest 2
zoom_3=[{20:120-1},{200:330-1}];  % Contest 3
zoom_1=[{690:790-1},{500:600-1}]; %show blue block
zoom_2=[{1340:1440-1},{920:1120-1}]; %show containers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate the blurring kernel and spectral response
fprintf('***Start estimating the degradation operators...***\n');
lambda_R = 1e1;
lambda_B = 1e1;
p=4; %p=20; for Qunming' data
intersection = cell(1,1);
overlap=1:p; % for Qunming' data % overlap=1:50; % overlap=1:48;
intersection{1} = overlap;
contiguous=intersection;
% Blur's support: [hsize_h hsize_w]
hsize_h = 11;
hsize_w = 11;
% hsize_h = 1; % for Qunming' data 
% hsize_w = 1;
shift = 0; % 'phase' parameter in MATLAB's 'upsample' function
blur_center = 0; % to center the blur kernel according to the simluated data
[~, R_est, B_est] = sen_resp_est(MS, PAN, ratio, intersection, contiguous,...
                p, lambda_R, lambda_B, hsize_h, hsize_w, shift, blur_center);
fprintf('***Finish estimating the degradation operators!***\n');

BluKer=MatrixToKernel(B_est,hsize_h,hsize_w);

figure(5);imagesc(BluKer);axis image;axis off;
set(gca,'FontSize',15);
colorbar;title('Estimated Spatial Blurring');

figure(6);plot(R_est','o-','LineWidth',2); 
 text([1 2-0.05 3-0.05 4-0.2],R_est+0.04,cellstr(num2str(round(R_est,5)')),'FontSize',15);
xlabel('MS bands','FontSize',15);
set(gca,'xtick',[1 2 3 4],'FontSize',15);
set(gca,'XTickLabel',{'NIR-1';'Red-2';'Green-3';'Blue-4'},'FontSize',15)
% ax = gca;ax.XTickLabel = {'NIR','Red','Green','Blue'};
title('Estimated Spectral Blurring');
drawnow;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('***Start fusing two images...***\n');
start_pos(1)=1; start_pos(2)=1; % The starting point of downsampling
%% Subspace identification
scale=1;[E_hyper, ~, ~, ~]=idHSsub(MS,'PCA',scale,p);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML estimator
tic
[I_ML] = BayesianFusion(MS,PAN,B_est,ratio,R_est,E_hyper,'ML',start_pos);
ML_time=toc;
tic
[I_BayesNaive]= BayesianFusion(MS,PAN,B_est,ratio,R_est,E_hyper,'Gaussian',start_pos);
Gaussian_time=toc;

% Compare with the results of ADMM
% tic
% [I_TV,Cost_set] = FuseTV(MS,PAN,B_est,ratio,R_est,E_hyper);
% TV_time=toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('***Finish fusing two images!***\n');
fprintf('***We use %4.2f seconds to restore a %4.0f*%4.0f*%1.0f image using ML estimator!*** \n',...
    ML_time,size(PAN,1),size(PAN,2),size(MS,3));
fprintf('***We use %4.2f seconds to restore a %4.0f*%4.0f*%1.0f image using Gaussian prior!*** \n',...
    Gaussian_time,size(PAN,1),size(PAN,2),size(MS,3));
% fprintf('***We use %4.2f seconds to restore a %4.0f*%4.0f*%1.0f image using TV prior!*** \n',...
%     TV_time,size(PAN,1),size(PAN,2),size(MS,3));
%% Plot the fused MS images
figure(3);imshow(normColor(I_ML(:,:,MSband_show)));
set(gca,'FontSize',15);title('Fusion ML');
% print(figure(3),'-dpng','E:\qwei2\Bayesian_fusion\figures\I_ML.png')
% print(figure(3),'-depsc','E:\qwei2\Bayesian_fusion\figures\I_ML.eps')
figure(4);imshow(normColor(I_BayesNaive(:,:,MSband_show)));%title('Fusion');
set(gca,'FontSize',15);title('Fusion Bayes');
drawnow;

figure(32);
% zoomed 1 part
subplot(2,2,1);imshow(normColor(MS(round(zoom_1{1}/4),round(zoom_1{2}/4),MSband_show)));title('MS','FontSize',15);
subplot(2,2,2);imshow(normColor(PAN(zoom_1{1},zoom_1{2},:))); title('PAN','FontSize',15);
subplot(2,2,3);imshow(normColor(I_ML(zoom_1{1},zoom_1{2},MSband_show)));title('ML','FontSize',15);
subplot(2,2,4);imshow(normColor(I_BayesNaive(zoom_1{1},zoom_1{2},MSband_show)));title('Bayes','FontSize',15);
% subplot(3,2,5);imshow(normColor(I_TV(zoom_1{1},zoom_1{2},MSband_show)));

figure(33);
% zoomed 2 part
subplot(2,2,1);imshow(normColor(MS(round(zoom_2{1}/4),round(zoom_2{2}/4),MSband_show)));title('MS','FontSize',15);
subplot(2,2,2);imshow(normColor(PAN(zoom_2{1},zoom_2{2},:)));title('PAN','FontSize',15);
subplot(2,2,3);imshow(normColor(I_ML(zoom_2{1},zoom_2{2},MSband_show)));title('ML','FontSize',15);
subplot(2,2,4);imshow(normColor(I_BayesNaive(zoom_2{1},zoom_2{2},MSband_show)));title('Bayes','FontSize',15);
% subplot(3,2,5);imshow(normColor(I_TV(zoom_2{1},zoom_2{2},MSband_show)));

figure(34);
% details for zoomed 1 part
subplot(4,4,2);imshow(normColor(MS(round(zoom_1{1}/4),round(zoom_1{2}/4),MSband_show)));title('MS');
subplot(4,4,1);imshow(normColor(PAN(zoom_1{1},zoom_1{2},:)));title('PAN');
subplot(4,4,3);imshow(normColor(I_ML(zoom_1{1},zoom_1{2},MSband_show)));title('ML');
subplot(4,4,4);imshow(normColor(I_BayesNaive(zoom_1{1},zoom_1{2},MSband_show)));title('Bayes');
subplot(4,4,5); imshow(MS(round(zoom_1{1}/4),round(zoom_1{2}/4),1),[]);colorbar;caxis([0 0.5]);title('NIR');ylabel('MS');
subplot(4,4,6);imshow(MS(round(zoom_1{1}/4),round(zoom_1{2}/4),2),[]);colorbar;caxis([0 0.5]);title('Red');
subplot(4,4,7);imshow(MS(round(zoom_1{1}/4),round(zoom_1{2}/4),3),[]);colorbar;caxis([0 0.5]);title('Green');
subplot(4,4,8);imshow(MS(round(zoom_1{1}/4),round(zoom_1{2}/4),4),[]);colorbar;caxis([0 0.5]);title('Blue');

subplot(4,4,9);imshow(I_ML(zoom_1{1},zoom_1{2},1),[]);colorbar;caxis([0 0.5]);ylabel('ML');
subplot(4,4,10);imshow(I_ML(zoom_1{1},zoom_1{2},2),[]);colorbar;caxis([0 0.5])
subplot(4,4,11);imshow(I_ML(zoom_1{1},zoom_1{2},3),[]);colorbar;caxis([0 0.5])
subplot(4,4,12);imshow(I_ML(zoom_1{1},zoom_1{2},4),[]);colorbar;caxis([0 0.5])

subplot(4,4,13);imshow(I_BayesNaive(zoom_1{1},zoom_1{2},1),[]);colorbar;caxis([0 0.5]);ylabel('Bayes');
subplot(4,4,14);imshow(I_BayesNaive(zoom_1{1},zoom_1{2},2),[]);colorbar;caxis([0 0.5]);
subplot(4,4,15);imshow(I_BayesNaive(zoom_1{1},zoom_1{2},3),[]);colorbar;caxis([0 0.5]);
subplot(4,4,16);imshow(I_BayesNaive(zoom_1{1},zoom_1{2},4),[]);colorbar;caxis([0 0.5]);

figure(35);
% details for zoomed 2 part
subplot(4,4,1);imshow(normColor(MS(round(zoom_2{1}/4),round(zoom_2{2}/4),MSband_show)));title('MS');
subplot(4,4,2);imshow(normColor(PAN(zoom_2{1},zoom_2{2},:)));title('PAN');
subplot(4,4,3);imshow(normColor(I_ML(zoom_2{1},zoom_2{2},MSband_show)));title('ML');
subplot(4,4,4);imshow(normColor(I_BayesNaive(zoom_2{1},zoom_2{2},MSband_show)));title('Bayes');

subplot(4,4,5); imshow(MS(round(zoom_2{1}/4),round(zoom_2{2}/4),1),[]);colorbar;caxis([0 0.5]);title('NIR');ylabel('MS');
subplot(4,4,6);imshow(MS(round(zoom_2{1}/4),round(zoom_2{2}/4),2),[]);colorbar;caxis([0 0.5]);title('Red');
subplot(4,4,7);imshow(MS(round(zoom_2{1}/4),round(zoom_2{2}/4),3),[]);colorbar;caxis([0 0.5]);title('Green');
subplot(4,4,8);imshow(MS(round(zoom_2{1}/4),round(zoom_2{2}/4),4),[]);colorbar;caxis([0 0.5]);title('Blue');

subplot(4,4,9);imshow(I_ML(zoom_2{1},zoom_2{2},1),[]);colorbar;caxis([0 0.5]);ylabel('ML');
subplot(4,4,10);imshow(I_ML(zoom_2{1},zoom_2{2},2),[]);colorbar;caxis([0 0.5])
subplot(4,4,11);imshow(I_ML(zoom_2{1},zoom_2{2},3),[]);colorbar;caxis([0 0.5])
subplot(4,4,12);imshow(I_ML(zoom_2{1},zoom_2{2},4),[]);colorbar;caxis([0 0.5])

subplot(4,4,13);imshow(I_BayesNaive(zoom_2{1},zoom_2{2},1),[]);colorbar;caxis([0 0.5]);ylabel('Bayes');
subplot(4,4,14);imshow(I_BayesNaive(zoom_2{1},zoom_2{2},2),[]);colorbar;caxis([0 0.5]);
subplot(4,4,15);imshow(I_BayesNaive(zoom_2{1},zoom_2{2},3),[]);colorbar;caxis([0 0.5]);
subplot(4,4,16);imshow(I_BayesNaive(zoom_2{1},zoom_2{2},4),[]);colorbar;caxis([0 0.5]);

% figure(100);%title('Cost function')
% semilogy(Cost_set,'b-','LineWidth',2);
% xlim([1 length(Cost_set)]);
% set(gca,'FontSize',13);
% xlabel('Iteration number','FontSize',18);
% ylabel('Cost function','FontSize',18);
% print(figure(100),'-depsc','E:\qwei2\Bayesian_fusion\figures\Cost.eps')
