function   x    =   H_z(z, fft_B, sf, sz)
[ch, n]         =    size(z);

%  s0                       =    floor(sf/2);
  s0                       =    1;

if ch==1
    Hz          =    real( ifft2(fft2( reshape(z, sz) ).*fft_B) );
    x           =    Hz(1:sf:end, 1:sf:end);
    x           =    (x(:))';
else
    x           =    zeros(ch, floor(n/(sf^2)));    
    for  i  = 1 : ch
        Hz         =    real( ifft2(fft2( reshape(z(i,:), sz) ).*fft_B) );
        t          =    Hz(s0:sf:end, s0:sf:end);
        x(i,:)     =    (t(:))';
    end
end


