function  y    =    PTx( a,fft_B,s0,sf )
fft_BT=conj(fft_B);
z=zeros(sf*size(a,1),size(a,2));
z(s0:sf:end, :)     =    a;
y                           =    real( ifft(fft( z ).*repmat(fft_BT,1, size(z,2))) );
