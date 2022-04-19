function  y    =    A_x( x, mu, fft_B, fft_BT, sf, sz )
z                           =    zeros(sz);
s0                          =    floor(sf/2);
Hx                          =    real( ifft2(fft2( reshape(x, sz) ).*fft_B) );
z(s0:sf:end, s0:sf:end)     =    Hx(s0:sf:end, s0:sf:end);
y                           =    real( ifft2(fft2( z ).*fft_BT) );
y                           =    y(:) + mu*x;


