function [B]=KernelToMatrix(KerBlu,nr,nc)
% flip the kernel
KerBlu=rot90(KerBlu,2);
mid_col=round((nc+1)/2);
mid_row=round((nr+1)/2);
% the size of the kernel
[len_hor,len_ver]=size(KerBlu);
lx = (len_hor-1)/2;
ly = (len_ver-1)/2;
B=zeros(nr,nc);
% range of the pixels
B(mid_row-lx:mid_row+lx,mid_col-ly:mid_col+ly)=KerBlu;
B=circshift(B,[-mid_row+1,-mid_col+1]);