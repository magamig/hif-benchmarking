function   z    =   HT_y(y, fft_BT, sf, sz)
[ch,n]                   =    size(y);
 s0                       =    floor(sf/2);
  s0                       =    1; 
if  ch  == 1
    z                        =    zeros(sz);    
    z(s0:sf:end, s0:sf:end)  =    reshape(y, floor(sz/sf));
    z                        =    real( ifft2(fft2( z ).*fft_BT) );
    z                        =    z(:)';
else    
    z                        =    zeros(ch, n*sf^2);
    t                        =    zeros(sz);
    for  i  = 1 : ch
        t(s0:sf:end,s0:sf:end)        =    reshape(y(i,:), floor(sz/sf));
        Htz                           = real( ifft2(fft2( t ).*fft_BT) );
        z(i,:)                        =    Htz(:)';
    end
end

