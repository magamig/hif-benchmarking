function par = ParSet_new(sf,sz,kernel_type)
%sf: scaling factor
%sz: spatial resolution 


if nargin < 3
%     sz = [512,512] ;
    kernel_type = 'Uniform_blur';       % Uniform_blur as default
end

if strcmp(kernel_type, 'Uniform_blur')
    psf        =    ones(sf)/(sf^2);
elseif strcmp(kernel_type, 'Gaussian_blur')
    psf        =    fspecial('gaussian',8,3);
end

%% Weisheng Dong Eq.19

par.fft_B      =    psf2otf(psf,sz);
par.fft_BT     =    conj(par.fft_B);

par.H          =    @(z)H_z(z, par.fft_B, sf, sz );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz);

%% Parameters of FBP and block matching
par.patsize = 4;        % main_SM's best parameter is 7&6, main_new is 4&2 
par.Pstep = 2;
par.patnum  = 60;               
par.step          =   floor((par.patsize-1));   
par.nCluster = 300;

%% algorithm 1  
par.eta = 0.1;
par.lambda = 0.001;     % fixed

if strcmp(kernel_type, 'Uniform_blur')
    if sf == 8
%         par.eta = 0.05;     % under opt
        par.iter = 100;
        par.mu = 0.0004;    %0.0004 at first
        par.rho = 1.05;     %1.05 at first
    elseif sf ==16
        par.iter = 200;
        par.mu = 0.0001;
        par.rho = 1.03;
    elseif sf ==32
        par.iter = 300;
        par.mu = 0.00005;
        par.rho = 1.005;
    end
elseif strcmp(kernel_type,'Gaussian_blur')
    par.eta = 0.004;
    par.iter = 100;
    par.mu = 0.0004;
    par.rho = 1.05;
end