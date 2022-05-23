% gti_path="data/GT/CAVE/balloons_ms.mat";scale=4;sri_path="data/paper_SR/GSOMP/CAVE/4/balloons_ms.mat";GSOMP_paper_run;

GTI = im2double(load(gti_path).hsi);
[M,N,L] = size(GTI);

T = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004 0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0  
    0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003
    0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];

MSI = hyperConvert3D((T*hyperConvert2D(GTI)), M, N);
[HSI] = downsample(GTI, scale);

[rows1,cols1,bands1] = size(MSI);
[rows2,cols2,bands2] = size(HSI);

maxval = max([max(HSI(:)) max(MSI(:))]);
if maxval > 1
    HSI = HSI/maxval;
    MSI = MSI/maxval;
end

param.spams = 1;        % Set = 1 if SPAMS***(see below)is installed, 0 otherwise
param.L = 20;           % Atoms selected in each iteration of G-SOMP+
param.gamma = 0.99;     % Residual decay parameter
param.k = bands2;          % Number of dictionary atoms
param.eta = 10e-3;      % Modeling error
patchsize = rows2;

[Out] = superResolution2(param,HSI,MSI,T,patchsize);
if maxval > 1
    Out = Out * maxval;
end

sri = im2uint16(Out);

save(sri_path, 'sri');