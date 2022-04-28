% gti_path="data/GT/CAVE/balloons_ms.mat";scale=4;sri_path="data/paper_SR/FUSE/CAVE/4/balloons_ms.mat";FUSE_paper_run;

GTI = im2double(load(gti_path).hsi);
ratio = scale;

[M,N,L] = size(GTI);
scaling = 10000;

T = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004 0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0  
    0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003
    0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];

MSI = hyperConvert3D((T*hyperConvert2D(GTI)), M, N, 3);

size_kernel=[9 9];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; 
start_pos(2)=1;
[HSI,KerBlu]=conv_downsample(GTI,ratio,size_kernel,sig,start_pos);

[Out]= BayesianFusion(HSI*scaling,MSI*scaling,T,KerBlu,ratio,'Gaussian',start_pos);
if mod(ratio,2) == 0
    start_pos = [round(ratio/2)+1 round(ratio/2)+1];
    [Out2]= BayesianFusion(HSI*scaling,MSI*scaling,T,KerBlu,ratio,'Gaussian',start_pos);
    Out = (Out+Out2)/2;
end
sri = Out/scaling;

sri = im2uint16(sri);

save(sri_path, 'sri');