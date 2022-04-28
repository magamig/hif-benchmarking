%   Figures for paper

%  Figure 1
%   from testDictionaryLearning(1)
% dim = 64; nAtoms = 192; k = 6; nSignals = 50,000; avgAtoms = k * nSignals / nAtoms;
% M_batch_OMP=dim*nAtoms^2 + nSignals * [ dim * ( 1 + nAtoms) + k^2 * ( 1 + nAtoms + k^2 ) ];
% m_D_update = nAtoms * [ 2  + dim * ( 3 + nAtoms ) + avgAtoms * ( 2 + nAtoms) ]

load DUCs8

%   Number of multiplications for batch_OMP = 1.03*10^9
%   Number of multiplications for optimize_atom = 6.06*10^7
time2DUCS = 1 + 6.06/103;
time4DUCS = 1 + 3*6.06/103;
figure; 
pl = plot(1:params.iternum,err1,'-',[1:params.iternum]*time2DUCS,err2,'-.',[1:params.iternum]*time4DUCS,err4,'--'); 
set(pl,'LineWidth',2.0,'MarkerSize',10)
xlabel('Computations','FontSize',18,'FontName','Times'); 
set(gca,'FontName','Times'); 
ylabel('RMSE','FontSize',18,'FontName','Times');
title('Representation RMSE - Training Data','FontSize',18,'FontName','Times');
lh=legend('1 DUC','2 DUCs','4 DUCs');
set(lh,'FontSize',18,'FontName','Times')
set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
print -depsc2 DUCsDiX8trainCPU.eps

%  Figure 2
%   from testDictionaryLearning(1)

figure; 
pl = plot(1:params.iternum,gerr1,'-',[1:params.iternum]*time2DUCS,gerr2,'-.',[1:params.iternum]*time4DUCS,gerr4,'--'); 
set(pl,'LineWidth',2.0,'MarkerSize',10)
xlabel('Computations','FontSize',18,'FontName','Times'); 
set(gca,'FontName','Times'); 
ylabel('RMSE','FontSize',18,'FontName','Times');
title('Representation RMSE - Testing Data','FontSize',18,'FontName','Times');
lh=legend('1 DUC','2 DUCs','4 DUCs');
set(lh,'FontSize',18,'FontName','Times')
set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
print -depsc2 DUCsDiX8testCPU.eps

%  Figure 3
% from testDictionaryLearning(3)
% load MODLSX8.mat
% 
% figure; 
% pl = plot(1:params.iternum,err1,'-',1:params.iternum,err2,'-.',1:params.iternum,err3,'--',1:params.iternum,err4,':'); 
% set(pl,'LineWidth',2.0,'MarkerSize',10)
% xlabel('Iteration','FontSize',22,'FontWeight','bold'); ylabel('RMSE','FontSize',22,'FontWeight','bold');
% lh = legend('K-SVD','2 DUCs','MOD-LS','2 DUCs + MOD-LS');
% set(lh,'FontSize',20,'FontWeight','bold')
% set(gca,'FontSize',22,'LineWidth',2.0,'FontWeight','bold')

%  Figure 4
%   from testDictionaryLearning(3)
% 
% figure; 
% pl = plot(1:params.iternum,gerr1,'-',1:params.iternum,gerr2,'-.',1:params.iternum,gerr3,'--',1:params.iternum,gerr4,':'); 
% set(pl,'LineWidth',2.0,'MarkerSize',10)
% xlabel('Iteration','FontSize',22,'FontWeight','bold'); ylabel('RMSE','FontSize',22,'FontWeight','bold');
% lh = legend('K-SVD','2 DUCs','MOD-LSX','MOD-LSX + 2 DUCs');
% set(lh,'FontSize',20,'FontWeight','bold')
% set(gca,'FontSize',22,'LineWidth',2.0,'FontWeight','bold')

%  Figure 5
%   from testDictionaryLearning(5.6)
% dim=225;nAtoms=675;k=23;nSignals=50000; avgIter=5.8;k3=round(k/3);
% M_batch_CoefROMP=dim*nAtoms^2 +nSignals * (dim*nAtoms +k3*(1+nAtoms +k3^2) + avgIter*k*(2*k^2 +nAtoms+1) );
% M_batch_CoefROMP= 1.955804562500000e+10
% M_batch_OMP=dim*nAtoms^2 + nSignals * [ dim * ( 1 + nAtoms) + k^2 * ( 1 + nAtoms + k^2 ) ];
% M_batch_OMP=3.957976562500000e+10
% m_D_update = nAtoms * [ 2  + dim * ( 3 + nAtoms ) + avgAtoms * ( 2 + nAtoms) ]
% M_Total_OMP =    4.0461e+10
% M_Total_CoefROMP =   2.0440e+10

load CoROMPDiX15

timeCoefROMP = 2.044 / 4.0461;
figure; 
pl = plot(1:params.iternum,err1,'-',[1:params.iternum]*timeCoefROMP,err2,'-.'); 
set(pl,'LineWidth',2.0,'MarkerSize',10)
xlabel('Computations','FontSize',18,'FontName','Times'); 
ylabel('RMSE','FontSize',18,'FontName','Times');
title('Representation RMSE - Training Data','FontSize',18,'FontName','Times');
lh=legend('OMP','CoefROMP');
set(lh,'FontSize',18,'FontName','Times')
set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
print -depsc2 CoROMP15trainCPU.eps

%  Figure 6
%   from testDictionaryLearning(5.6)

figure; 
pl = plot(1:params.iternum,gerr1,'-',[1:params.iternum]*timeCoefROMP,gerr2,'-.'); 
set(pl,'LineWidth',2.0,'MarkerSize',10)
xlabel('Computations','FontSize',18,'FontName','Times'); 
ylabel('RMSE','FontSize',18,'FontName','Times');
title('Representation RMSE - Testing Data','FontSize',18,'FontName','Times');
lh=legend('OMP','CoefROMP');
set(lh,'FontSize',18,'FontName','Times')
set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
print -depsc2 CoROMP15testCPU.eps

%  Figure 7
%   from denoiseDemo2(2.6)
% load iterDUCS8
% 
% figure; 
% pl = plot(1:nIters,psnr(:,1),'-.',1:nIters,psnr(:,2),'-');
% set(pl,'LineWidth',2.0,'MarkerSize',10);
% title( ['PSNR for ' num2str(bb) '-' num2str(img) '-' num2str(sigma)],'FontSize',22,'FontWeight','bold');
% xlabel('Iteration','FontSize',22,'FontWeight','bold'); ylabel('PSNR','FontSize',22,'FontWeight','bold');
% lh=legend('1 DUC','4 DUCs');
% set(lh,'FontSize',20,'FontWeight','bold')
% set(gca,'FontSize',22,'LineWidth',2.0,'FontWeight','bold')
% 
% %  Figure 8
% %   from denoiseDemo2(2.6)
% load iterDUCS16
% 
% figure; 
% pl = plot(1:nIters,psnr(:,1),'-.',1:nIters,psnr(:,2),'-');
% set(pl,'LineWidth',2.0,'MarkerSize',10);
% title( ['PSNR for ' num2str(bb) '-' num2str(img) '-' num2str(sigma)],'FontSize',22,'FontWeight','bold');
% xlabel('Iteration','FontSize',22,'FontWeight','bold'); ylabel('PSNR','FontSize',22,'FontWeight','bold');
% lh=legend('1 DUC','4 DUCs');
% set(lh,'FontSize',20,'FontWeight','bold')
% set(gca,'FontSize',22,'LineWidth',2.0,'FontWeight','bold')


% Figure 9
%  from denoiseDemo2(1.5)
% load nAtomDUC8-1-25.mat     %   As in paper
load iterDUC8-1-25   

figure;
pl = plot(1:nIters,psnr(:,1,2),'-+',1:nIters,psnr(:,2,2),'-o', ...
1:nIters,psnr(:,1,1),'-.+',1:nIters,psnr(:,2,1),'-.o');
set(pl,'LineWidth',2.0,'MarkerSize',10);
ylabel('PSNR','FontSize',18,'FontName','Times'); 
xlabel('Iteration','FontSize',18,'FontName','times');
title('Denoising Performance for 8\times 8 Patches','FontSize',18,'FontName','Times');
lh = legend('CoefROMP; 1 DUC','CoefROMP; 2 DUC','OMP; 1 DUC','OMP; 2 DUC',4);
set(lh,'FontSize',14)
set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
print -depsc2 PSNR8-25Barbara.eps

% Figure 10
%   from denoiseDemo2(1.5)  for the 16x16 case
load iterDUC16-1-25   

figure; 
pl = plot(1:nIters,psnr(:,1,2),'-+',1:nIters,psnr(:,2,2),'-o', ...
    1:nIters,psnr(:,1,1),'-.+',1:nIters,psnr(:,2,1),'-.o');
set(pl,'LineWidth',2.0,'MarkerSize',10);
ylabel('PSNR','FontSize',18,'FontName','Times'); 
xlabel('Iteration','FontSize',18,'FontName','times');
title('Denoising Performance for 16\times 16 Patches','FontSize',18,'FontName','Times');
lh = legend('CoefROMP; 1 DUC','CoefROMP; 2 DUC','OMP; 1 DUC','OMP; 2 DUC',4);
set(lh,'FontSize',14)
set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
print -depsc2 PSNR16-25Barbara.eps


% load nAtomIter8-1-25.mat
% rng_nAtoms = [0.75 1 2 4 6 8 12 16];  % *64 = dictsize
% 
% figure;surf(psnr(:,:,2)-psnr(:,:,1))
% title('PSNR improvement using CoROMP','FontSize',18);
% ylabel('Iteration','FontSize',18,'FontWeight','bold');zlabel('PSNR','FontSize',18,'FontWeight','bold');xlabel('Number of atoms','FontSize',18,'FontWeight','bold')
% % set(gca,'XTickLabelMode','manual','XTickLabel',{'0'; '48'; '64';'128'; '256';'384';'512';'768';'1024'})
% set(gca,'FontSize',22,'LineWidth',2.0,'FontWeight','bold')
% 
