% gti_path="data/GT/CAVE/balloons_ms.mat";scale=4;sri_path="data/paper_SR/CNNFUS/CAVE/4/balloons_ms.mat";CNNFUS_paper_run;

rand('seed',0);

S = im2double(load(gti_path).hsi); % GTI
[M,N,L] = size(S);
S_bar = hyperConvert2D(S);
sz = [M,N];
sf = scale;

psf = fspecial('gaussian',7,2);
fft_B = psf2otf(psf,sz);

F=create_F();
for band = 1:size(F,1)
    div = sum(F(band,:));
    for i = 1:size(F,2)
        F(band,i) = F(band,i)/div;
    end
end

H = @(z)H_z(z, fft_B, sf, sz, 1);
hyper = H(S_bar);
Y_h = hyperConvert3D(hyper, M/sf, N/sf);
Y = hyperConvert3D((F*S_bar), M, N);

para.gama=1.1;
para.p=10;
para.sig=10e-4;

[Out]= CNN_Subpace_FUS(Y_h,Y,F,fft_B,sf,nan,para,1);

sri = im2uint16(Out);

save(sri_path, 'sri');
