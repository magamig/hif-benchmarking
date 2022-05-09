function REC = reconstruct_images(Z,Y,dims,dsf,roi,fst,saveImages)
% Reconstruct Images from Variables
Z_IMGS = zeros(dims(1)*dsf,dims(2)*dsf,31);
REC.NOISE = 100*(norm(Y - create_F()*Z,'fro')^2)/(norm(Y,'fro')^2);
for band = 1:31    
    for row = 1:dims(1)*dsf
        for col = 1:dims(2)*dsf
            Z_IMGS(row,col,band) = Z(band,col + dims(2)*dsf*(row-1));
        end
    end
end
TEMPORARY_C_IMGS = zeros(dims(1)*dsf,3*dims(2)*dsf,31);
%REC.C_IMGS = zeros(dims(1)*dsf,3*dims(2)*dsf,31);
REC.RMSE = 0;
[Z_GT_IMGS scalar] = load_Z_imgs(fst,roi,dsf);
for band = 1:31
    TEMPORARY_C_IMGS(:,:,band) = [Z_GT_IMGS(:,:,band) Z_IMGS(:,:,band) abs(Z_GT_IMGS(:,:,band) - Z_IMGS(:,:,band))];
    for row = 1:dims(1)*dsf
        for col = (2*dims(2)*dsf+1):(3*dims(2)*dsf)
            REC.RMSE = REC.RMSE + TEMPORARY_C_IMGS(row,col,band)^2;
        end
    end 
end
bits = 16;
if scalar < 255
    bits = 8;
end
REC.RMSE = sqrt(REC.RMSE/(31*dims(2)*dsf*dims(1)*dsf))*255;
REC.PSNR = get_PSNR(TEMPORARY_C_IMGS,(2^bits - 1)/scalar);
if saveImages == 1
    REC.C_IMGS = TEMPORARY_C_IMGS;
end
end

