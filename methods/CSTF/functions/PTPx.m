function  y    =    PTPx( a,fft_B,s0,sf )
z                           =    zeros(size(a));
fft_BT=conj(fft_B);
Hx                          =    real( ifft(fft(  a).*repmat(fft_B,1, size(a,2))) );
z(s0:sf:end, :)     =    Hx(s0:sf:end, :);
y                           =    real( ifft(fft( z ).*repmat(fft_BT,1, size(z,2))) );
