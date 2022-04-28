% gti_path="data/GT/CAVE/balloons_ms.mat";scale=4;sri_path="data/paper_SR/NSSR/CAVE/4/balloons_ms.mat";NSSR_paper_run;

rand('seed',0);

S = im2double(load(gti_path).hsi); % GTI
[M,N,L] = size(S);
S_bar = hyperConvert2D(S);
sz = [M,N];
sf = scale;

s0=1;
psf        =    fspecial('gaussian',7,2);
par.fft_B      =    psf2otf(psf,sz);
par.fft_BT     =    conj(par.fft_B);
par.H          =    @(z)H_z(z, par.fft_B, sf, sz,s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, sf, sz,s0);
par.P=create_F();
F=create_F();
for band = 1:size(F,1)
    div = sum(F(band,:));
    for i = 1:size(F,2)
        F(band,i) = F(band,i)/div;
    end
end

hyper = par.H(S_bar);
Y_h = hyperConvert3D(hyper, M/sf, N/sf);
Y = hyperConvert3D((F*S_bar), M, N);

para.K=160;
para.eta=1e-2;

Out = LTTR_FUS(Y_h,Y,F,para.K,para.eta, par.fft_B,sf,S);

sri = im2uint16(Out);

save(sri_path, 'sri');
