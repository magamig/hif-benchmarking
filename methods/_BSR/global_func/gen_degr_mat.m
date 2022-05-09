function B=gen_degr_mat(blur_type,nr,nc,lx,ly)
sx = 2.5;
sy = 2.5;
% sx = 0.5;
% sy = 0.5;
% 
%
% define convolution operator
%

mid_col=round((nc+1)/2);
mid_row=round((nr+1)/2);

% lx = 1;
% ly = 1;

B=zeros(nr,nc);

if lx > 0
    for i=-ly:ly
        for j=-lx:lx
            if blur_type == 1;
                B(mid_row+i,mid_col+j)= 1;
            else
                B(mid_row+i,mid_col+j)=  exp(-((i/sy)^2 +(j/sx)^2)/2);
            end
        end
    end
else
    B(mid_row,mid_col) = 1;
end
%circularly center
%H=fftshift(H);
B=circshift(B,[-mid_row+1,-mid_col+1]);
%normalize
B=B/sum(sum(B));