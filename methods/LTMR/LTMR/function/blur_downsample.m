function   X   =   blur_downsample(YY,lambda, FH, d, sz)
M=sz(1);
N=sz(2);
OpCH = @(X) reshape(sum(im2col(X,[M/d,N/d],'distinct'),2) ,m,n);
OpC = @(X) repmat(X,d,d);
FHC = conj(FH);
 IF = 1./(OpCH(abs(FH).^2)/lambda + d^2);

X = YY/lambda -1/lambda^2*ifft2(FHC.*OpC(IF.*OpCH(FH.*fft2(YY))));