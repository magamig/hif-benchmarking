clear all;
clc;
res_path = './results/CAVE0615';
ori_path = '/media/lab-titan2/新加卷3/zc/0results/0result/ori_data/CAVE';
oris = dir(ori_path);
oris = oris(3:end);
ress = dir(res_path);
ress = ress(3:end);
mean_psnr  =   0 ;
mean_rmse  =   0 ; 
mean_sam   =   0 ;
mean_ssim  =   0 ;
mean_ergas =   0 ;
N_imgs = length(ress);
for i=1:N_imgs
    disp(oris(i).name)
    % ori
    hsi = zeros(512,512,31);
    for j=1:31
        name = strcat(oris(i).name,sprintf('_%02d.png',j));
        tmp = imread(fullfile(ori_path,oris(i).name,oris(i).name,name));
        if strcmp(oris(i).name,'watercolors_ms')
            hsi(:,:,j)=tmp(:,:,1);
        else
            hsi(:,:,j)=tmp;
        end
    end
    
    if strcmp(oris(i).name,'watercolors_ms')
        hsi = double(hsi)/255.;
    else
        hsi = double(hsi)/65535.;
    end
    %res
    load(fullfile(res_path,ress(i).name));
    res = permute(double(squeeze(res)),[2,3,1]);
%     res = res;
    [psnr,rmse, ergas, sam, ssim] = quality_assessment(hsi, res, 0, 1/8);
    mean_psnr  =   mean_psnr + psnr/N_imgs  ;
    mean_rmse  =   mean_rmse + rmse/N_imgs  ; 
    mean_sam   =   mean_sam  + sam/N_imgs   ;
    mean_ssim  =   mean_ssim + ssim/N_imgs  ;
    mean_ergas =   mean_ergas+ ergas/N_imgs ;
    disp(sprintf(' PSNR : %2.3f, RMSE : %2.5f, ERGAS : %2.5f, SAM : %2.5f, SSIM : %2.5f\n',psnr,rmse,ergas,sam,ssim));
end
 disp(sprintf('The Mean PSNR : %2.3f, RMSE : %2.3f, ERGAS : %2.3f, SAM : %2.3f, SSIM : %2.3f \n',mean_psnr,mean_rmse,mean_ergas,mean_sam,mean_ssim));


