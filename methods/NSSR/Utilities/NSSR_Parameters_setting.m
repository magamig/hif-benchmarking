function   par   =  NSSR_Parameters_setting( sf, kernel_type, sz )
par.h          =    sz(1);
par.w          =    sz(2);
par.K          =    80;
if strcmp(kernel_type, 'Uniform_blur')
    psf        =    ones(sf)/(sf^2);
elseif strcmp(kernel_type, 'Gaussian_blur')
    psf        =    fspecial('gaussian',8,3);
end
par.fft_B      =    psf2otf(psf,sz);
par.fft_BT     =    conj(par.fft_B);
par.H          =    @(z)H_z(z, par.fft_B, sf, sz );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz);
par.lambda     =    0.001; 

if strcmp(kernel_type,'Uniform_blur')
    if  sf==8
        par.eta1       =   0.015;    % 0.008
        par.eta2       =   0.0001;   % 0.00005;
        par.mu         =   0.0002;   % 0.0002;        
        par.ro         =   1.1;      % 1.09
        par.Iter       =   26;
    elseif sf==16
        par.eta1       =   0.015; 
        par.eta2       =   0.00008;
        par.mu         =   0.0001; 
        par.ro         =   1.1;        
        par.Iter       =   25;   % 45
    else
        par.eta1       =   0.025;   %0.008
        par.eta2       =   0.00008;
        par.mu         =   0.0001;  %0.002
        par.ro         =   1.1;         
        par.Iter       =   25;
    end    

elseif strcmp(kernel_type,'Gaussian_blur')
    par.eta1       =   0.015;    % 0.03
    par.eta2       =   0.00006;
    par.mu         =   0.0002;   % 0.004
    par.ro         =   1.1; 
    par.Iter       =   26;
end
    