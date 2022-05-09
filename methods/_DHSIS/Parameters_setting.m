function   par   =  Parameters_setting( sf, kernel_type, sz )
par.h          =    sz(1);
par.w          =    sz(2);

if strcmp(kernel_type, 'Uniform_blur')
    psf        =    ones(sf)/(sf^2);
elseif strcmp(kernel_type, 'Gaussian_blur')
    psf        =    fspecial('gaussian',8,2);
end

%   padSize = sz - [sf sf];
%    psf1     = padarray(psf, padSize, 'post');
%    B1   = circshift(psf1,-floor([2 2 ]/2));
%    B2=fftn(B1);
%    par.fft_B=B2;
par.psf=psf;
par.fft_B      =    psf2otf(psf,sz);
% par.fft_B =KernelToMatrix(KerBlu,nr,nc);

par.fft_BT     =    conj(par.fft_B);
par.H          =    @(z)H_z(z, par.fft_B, sf, sz );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz);



    