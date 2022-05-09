clear
clc

addpath('function')
sf = 8;

aaaa1=strcat('.\data\hout\SE\out.mat');
im_structure1 =load(aaaa1);
S=im_structure1.gt_out;
X=im_structure1.net_out;
S = double(S);
X = double(X);

PSNR = zeros(12,1);
SSIM = zeros(12,1);
SAM = zeros(12,1);
ERGAS = zeros(12,1);
RMSE = zeros(12,1);

psnr = 0;
ssim = 0;
sam = 0;
ergas = 0;
rmse = 0;
           
for i = 1:12
    i
    temp = S(i,:,:,:);
    temp1 = X(i,:,:,:);
    temp = reshape(temp, 512,512,31);
    temp1 = reshape(temp1, 512,512,31);
    [psnr1,rmse1, ergas1, sam1, uiqi1,ssim1] = quality_assessment(double(im2uint8(temp)), double(im2uint8(temp1)), 0, 1.0/sf);
    PSNR(i) = psnr1;
    SSIM(i) = ssim1;
    SAM(i) = sam1;
    ERGAS(i) = ergas1;
    psnr = psnr1/12 + psnr;
    ssim = ssim1/12 + ssim;
    sam = sam1/12 + sam;
    ergas = ergas1/12 + ergas;
    rmse = rmse1/12 + rmse;
end

save('.\data\CAVE.mat', 'PSNR', 'SSIM','SAM','ERGAS')


