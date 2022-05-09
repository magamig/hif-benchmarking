clear all; close all; 


rootpath = 'F:\ADSC_desktop\Research\Matlab codes\multspectral_imaging_data\'; %the path to
%the image database
dsf = 8;  %set the downsampling factor to 32
roi = [1 64 1 64]; %set the region of interest, interpret as [x1 x2 y1 y2]
%note: the region of interest corresponds to the region of the downsampled
%images, so for images originally 512 by 512 which are downsampled by a
%factor of 32, an roi of [1 16 1 16] corresponds to the entire image
N = 10; %set the number of endmembers
imagelist = {'watercolors_ms'}; %set which images you want to reconstruct
methodlist = {'SNNMF'}; 
rescale = 1; %no rescaling of the X data.  set to 0.9, or 1.1, for example,
%if you want to see the effect of a scale-ambiguity on the results
saveImages = 0; %saves fully reconstructed images of the scenes, in addition
%to the RMSE and PSNR information.  set to 0 if you don't want this in
%order to save memory.  note, if you save images, they will be called
%C_IMGS inside the results variable.  call the function preview(XXX.C_IMGS)
%to scroll through them one by one (by pressing any key)

%run experiments
filestem = char(strcat(rootpath,imagelist,'/',imagelist,'/',strrep(imagelist,'ms','')));

A = AVMAX(create_X(load_Z(filestem,roi,dsf),create_G(roi,dsf),roi, dsf, rescale),N);
SNNMF_OUT = SNNMF(filestem,roi,dsf,N,A,rescale,saveImages);