% gti_path="data/GT/CAVE/balloons_ms.mat";scale=4;sri_path="data/paper_SR/NSSR/CAVE/4/balloons_ms.mat";NSSR_paper_run;

rand('seed',0);

Z_ori = im2double(load(gti_path).hsi); % GTI
[M,N,L] = size(Z_ori);
sz = [M,N];
Z_ori = hyperConvert2D(Z_ori);
sf = scale;

par             =    NSSR_Parameters_setting( sf, 'Gaussian_blur', sz );
X               =    par.H(Z_ori);
par.P           =    create_P();
Y               =    par.P*Z_ori;

D               =    Nonnegative_DL( X, par );   
D0              =    par.P*D;
NLM             =    Comp_NLM_Matrix( Y, sz );   

Out         =    Nonnegative_SSR( D, D0, X, Y, NLM, par, Z_ori, sf, sz );
Out         =    reshape(transpose(Out),[M,N,L]);

sri = im2uint16(Out);

save(sri_path, 'sri');
