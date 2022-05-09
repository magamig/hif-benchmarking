close all;
addpath([pwd,'\func_CloseForm']);
normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3.5+0.5;
%% This script is to implement the joint fusion and unmixing of multi-band images
%If you use this code, please cite the following paper:

% [1] Q. Wei, J. M. Bioucas-Dias, N. Dobigeon, J-Y. Tourneret, M. Chen and S. Godsill,
% Multi-band image fusion based on spectral unmixing, IEEE Trans. Geosci. and
% Remote Sens., to appear.

%% -------------------------------------------------------------------------
% Copyright (June, 2016):        Qi WEI (qi.wei@eng.cam.ac.uk)
%
% EEA_Fusion is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
%% Part 1: Bulid the groud-truth, HS and MS images
%% Prepare the groudtruth image
SNR_HS=40;SNR_MS=40;   % SNR of HS and MS images
global E_real A_real X_real
%% Semi-Real Image
load('EndmembersFractal.mat');
load('fractal1.mat')
E_real=F2(:,1:9);
N_HS=size(E_real,1);
[nr,nc,L]=size(verdadTerreno(:,:,1:9));
sizeIM=[nr,nc,L];N=nr*nc;
A_real=reshape(verdadTerreno(:,:,1:9),N,L)';
% A_real=A_real./repmat(sum(A_real,1),[L 1]); % Normalize the abundances
VX_real=E_real*A_real;
X_real=reshape(VX_real',[nr nc N_HS]);
band_set=[20 11 4];
clear endmembersGT abundanciesGT syntheticImage

ImA_real=reshape(A_real',sizeIM);
figure(1);plot(E_real);title('ground truth endmembers')
temp_show=X_real(:,:,band_set);temp_show=normColor(temp_show);
figure(2);imshow(temp_show);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The degradation for HS image
% blur_type=2;
psfY.ds_r=4;
sig = 1.7;  KerBlu = fspecial('gaussian',[7 7],sig);
psfY.B=KernelToMatrix(KerBlu,nr,nc);
n_dr=nr/psfY.ds_r;     n_dc=nc/psfY.ds_r;

mask=zeros(size(psfY.B));
mask(1:psfY.ds_r:end,1:psfY.ds_r:end,:)=1;
psfY.dsp=mask;
%% Generate the HS image: Spatial degradation
ImA_hyper = func_blurringY(ImA_real,psfY).*repmat(psfY.dsp,[1 1 L]); % blurring and downsampling
ImA_hyper = ImA_hyper(1:psfY.ds_r:end,1:psfY.ds_r:end,:);
A_hyper=reshape(ImA_hyper,[size(ImA_hyper,1)*size(ImA_hyper,2) L])';
XH=reshape((E_real*A_hyper)',[size(ImA_hyper,1) size(ImA_hyper,2) N_HS]);

TemPow=XH(1:psfY.ds_r:end,1:psfY.ds_r:end,:).^2;
Ps = mean(TemPow(:));
sigma2y_real = Ps.*(10.^(-SNR_HS/10));  %Caclulate the noise power
XH = XH + randn(size(XH))*sqrt(sigma2y_real);
VXH=reshape(XH,[n_dr*n_dc N_HS])';
%% The intialization of Endmember matrix
[E_ini, ~, ~] = svmax(VXH,'endmembers',L); 
E_ini(E_ini<0) = 0;
E_ini(E_ini>1) = 1;
%% Generate the MS image: Spectral mixture
N_MS=4;psfZ=zeros(N_MS,N_HS);
N_tem=52;
psfZ(1,1:round(N_tem/4))=ones;
psfZ(2,round(N_tem/4)+1:round(N_tem/2))=ones;
psfZ(3,round(N_tem/2):round(3*N_tem/4))=ones;
psfZ(4,round(3*N_tem/4):N_tem)=ones;
psfZ=psfZ./repmat(sum(psfZ,2),1,size(psfZ,2));

E_multi=psfZ*E_real; %% MS endmember
XM=reshape((E_multi*A_real)',[nr nc N_MS]);

sigma2z_real=sigma2y_real;
for i=1:N_MS
    XM(:,:,i) = XM(:,:,i) + randn(size(XM(:,:,i)))*sqrt(sigma2z_real);
end
VXM=reshape(XM,[nr*nc N_MS])';
%% Part 2: different estimators for Endmember and Abundance matrices 
ChInv=diag(sigma2y_real)\eye(N_HS);
CmInv=diag(sigma2z_real)\eye(N_MS);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2.4: The joint fusion and unmixing [2016FUMI]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
para_opt_in.SpaDeg=1; %%  If the degradation B and S are known, SpaDeg=1; otherwise, SpaDeg=0
para_opt_in.N_it=3000;% The number of iterations for BCD updates
para_opt_in.thre_BCD=1e-4;
para_opt_in.E_ini=E_ini;
[Out_FUMI,para_opt_out]=JointFusionUnmix(VXH,VXM,ChInv,CmInv,psfY,psfZ,sizeIM,para_opt_in);
time.FUMI=toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[E_ini,indx] = best_permut_R(E_ini,E_real);
Out_FUMI.E_hyper=Out_FUMI.E_hyper(:,indx);
Out_FUMI.A_multi=Out_FUMI.A_multi(indx,:);
%% Plot the endmembers
figure(3);
subplot(3,1,1);plot(E_real);ylim([0 1]);title('Groundtruth');
subplot(3,1,2);plot(real(E_ini));ylim([0 1]);title('VCA');
subplot(3,1,3);plot(real(Out_FUMI.E_hyper));ylim([0 1]);title('FUMI');
%% Plot the Estimated Endmembers
load USGS_1995_Library.mat
wavelengths = datalib(:,1)*1000;
wavelengths = sort(wavelengths(1:N_HS));
band_remove=[];

band_keep=setdiff([1:length(wavelengths)],band_remove);
figure(4);
E_show=zeros(length(wavelengths),1);E_show(band_remove,:)=NaN;
E_show(band_keep,:)=E_real(:,1);
plot(wavelengths,E_show,'-k','LineWidth',2);
hold on;
E_show(band_keep,:)=E_ini(:,1);
plot(wavelengths,E_show,'-c','LineWidth',2);
E_show(band_keep,:)=Out_FUMI.E_hyper(:,1);
plot(wavelengths,E_show,'-b','LineWidth',2);
hold off;
ylabel('Reflectance','FontSize',18);
xlabel('Wavelength (nm)','FontSize',18);
xlim([wavelengths(1) wavelengths(end)])
set(gca,'linewidth',1,'fontsize',15);
legend('Groundtruth','Initialization','FUMI'); 

for i=2:L
    figure(i+400);
    E_show(band_keep,:)=E_real(:,i);
    plot(wavelengths,E_show,'-k','LineWidth',2);
    hold on;
    E_show(band_keep,:)=E_ini(:,i);
    plot(wavelengths,E_show,'-c','LineWidth',2);
    E_show(band_keep,:)=Out_FUMI.E_hyper(:,i);
    plot(wavelengths,E_show,'-b','LineWidth',2);
    hold off;
    ylabel('Reflectance','FontSize',18);
    xlabel('Wavelength (nm)','FontSize',18);
    xlim([wavelengths(1) wavelengths(end)])
    set(gca,'linewidth',1,'fontsize',15);
    legend('Groundtruth','Initialization','FUMI');    
end
%% Plot the fused image
X_FUMI=reshape((Out_FUMI.E_hyper*Out_FUMI.A_multi)',[nr nc N_HS]);
temp_show=X_FUMI(:,:,band_set);temp_show=normColor(temp_show);
err_FUMI=mean((X_real-X_FUMI).^2,3);
figure(5);imshow(temp_show);
norm_err=max(err_FUMI(:));
figure(6);imshow(err_FUMI/norm_err);colormap('default')
%% Plot the Estimated Abundance Mapps
ImA_multi_FUMI= reshape(Out_FUMI.A_multi',[nr nc L]);
for i=1:L    
   figure(100);subplot(3,3,i);imshow(ImA_real(:,:,i));axis off;axis equal; colormap gray
   figure(200);subplot(3,3,i);imshow(ImA_multi_FUMI(:,:,i));axis off;axis equal; colormap gray
end

SAM_end.FUMI  = mean(angBvec(E_real,Out_FUMI.E_hyper));
SAM_end.Ini  = mean(angBvec(E_real,E_ini));

NMSE_End.FUMI =20*log10(norm(Out_FUMI.E_hyper-E_real,'fro')/norm(E_real,'fro'));
NMSE_End.Ini  =20*log10(norm(E_ini-E_real,'fro')/norm(E_real,'fro'));

NMSE_Abu.FUMI=20*log10(norm(Out_FUMI.A_multi-A_real,'fro')/norm(A_real,'fro'));

disp('SAM_end: ');disp(SAM_end);
disp('MMSE_End: ');disp(NMSE_End);
disp('NMSE_Abu: ');disp(NMSE_Abu);

[~,~,~,SNR.FUMI,Q.FUMI,SAM.FUMI,RMSE_fusion.FUMI,ERGAS.FUMI,DD.FUMI] = metrics(X_real,X_FUMI,psfY.ds_r);
fprintf('Fusion Performance of FUMI:\n SNR: %f\n RMSE: %f\n UIQI: %f\n SAM: %f\n ERGAS: %f\n DD: %f\n',...
        SNR.FUMI,RMSE_fusion.FUMI,Q.FUMI,SAM.FUMI,ERGAS.FUMI,DD.FUMI);
disp(time);
