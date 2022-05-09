function [KerBlu]=MatrixToKernel(B,len_hor,len_ver)
% the size of the kernel
lx = (len_hor-1)/2; % rows
ly = (len_ver-1)/2; % columns
[nr,nc]=size(B);
mid_col=round((nc+1)/2);
mid_row=round((nr+1)/2);
% range of the pixels
% B=circshift(B,[mid_row-1,mid_col-1]); %% equivalent with fftshift
B=fftshift(B); % center the low-frequency component
KerBlu=B(mid_row-lx:mid_row+lx,mid_col-ly:mid_col+ly);
KerBlu=rot90(KerBlu,2);


