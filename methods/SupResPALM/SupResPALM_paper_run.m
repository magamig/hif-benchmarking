% gti_path="data/GT/CAVE/balloons_ms.mat";scale=4;sri_path="data/paper_SR/SupResPALM/CAVE/4/balloons_ms.mat";SupResPALM_paper_run;

rand('seed',0);

truth = im2double(load(gti_path).hsi); % GTI
[M,N,L] = size(truth);
truth = hyperConvert2D(truth);

srf = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004 0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0
    0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003
    0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];

[ hyper, multi ] = hyperSynthetic( truth, srf, scale );

p=10;
[E,A] = SupResPALM(hyper, multi, srf, p, M);

Out = hyperConvert3d(E*A);

sri = im2uint16(Out);

save(sri_path, 'sri');
