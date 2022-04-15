function p=gausspdf(hsi,m,C)
% あるクラスに各ピクセルが属する確率を求める
% p (nr,nc)
nb=size(hsi,1);
nr=size(hsi,2);
nc=size(hsi,3);

p=zeros(nr,nc);
Cdet=det(C);
if Cdet~=0
    %Cinv=inv(C);
    K=((2.*pi)^(-nb/2))*(Cdet^(-0.5));
    p=K*exp(-gdistance(hsi,m,C));
end

if p<=0
    %p=rand(1)*1e-5;
    p=0;
end
