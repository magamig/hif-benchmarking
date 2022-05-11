function  [Z] =  CNN_Subpace_FUS( HSI, MSI,R,FBm,sf,S,para,gama)
% gama=1;
global sigmas
filepath = fileparts(mfilename('fullpath'));
load([filepath '/../FFDNet-master/FFDNet-master/models/FFDNet_gray.mat']);
%   load('G:\HSI-superresolution\FFDNet-master\FFDNet-master\models\FFDNet_Clip_gray.mat');
sig=para.sig;
 %net = vl_simplenn_tidy(net);
%net=vl_simplenn_move(net, 'gpu') ;
p=para.p;
mu=1e-3;


HSI3=Unfold(HSI,size(HSI),3);
% [w Rw] = estNoise(HSI3,'additive');
% [~, D]=hysime(HSI3,w,Rw);
[D,~,~]=svds(HSI3,p);
D=D(:,1:p);

RD=R*D;
L1=size(D,2);
nr=size(MSI,1);
nc=size(MSI,2);



L=size(HSI,3);

HSI_int=zeros(nr,nc,L);
HSI_int(1:sf:end,1:sf:end,:)=HSI;
FBmC  = conj(FBm);
FBs  = repmat(FBm,[1 1 L]);
FBs1  = repmat(FBm,[1 1 L1]);
       FBCs=repmat(FBmC,[1 1 L]);
FBCs1=repmat(FBmC,[1 1 L1]);
HHH=ifft2((fft2(HSI_int).*FBCs));
  HHH1=hyperConvert2D(HHH);




%% iteration

MSI3=Unfold(MSI,size(MSI),3);

n_dr=nr/sf;
n_dc=nc/sf;

HR_load1=imresize(HSI, sf,'bicubic');

% V2=D'*hyperConvert2D(HR_load1);
V2=zeros(p,size(MSI,1)*size(MSI,2));


G2=zeros(size(V2));

CCC=(gama*RD'*MSI3+D'*HHH1);
%  [Q,Lambda]=eig(C1);
% Lambda=reshape(diag(Lambda),[1 1 L1]);
% InvLbd=1./repmat(Lambda,[ sf*n_dr  sf*n_dc 1]);
% B2Sum=PPlus(abs(FBs1).^2./( sf^2),n_dr,n_dc);
% InvDI=1./(B2Sum(1:n_dr,1:n_dc,:)+repmat(Lambda,[n_dr n_dc 1]));



for i=1:15
     C1=gama*(RD)'*RD+mu*eye(size(D,2)); 
    HR_HSI3=mu*(V2+G2/(2*mu));
C3=CCC+HR_HSI3;

% if i>=2
% A0=A;
% end
   [A] = Sylvester(C1,FBm, sf,n_dr,n_dc,C3);  
    
Zt=hyperConvert3D(D*A,nr, nc );
% aa1=C1*ZE-C3+ creat_HSI_T(creat_HSI(ZE ,psfY),psfY);
% norm(aa1(:))

%  if i>=2
%  V20=V2;
%  end

   %psnr(i)=csnr(double(im2uint8(S)),double(im2uint8(Zt)),0,0)
B2=A-G2/(2*mu);
 B2=hyperConvert3D(B2,nr, nc );
 V2=zeros(size(B2));
 


%  p= parpool('local',2);
%% CNN denoiser
for jj=1:size(B2,3)
    eigen_im=(  B2(:,:,jj));
    min_x = min(eigen_im(:));
    max_x = max(eigen_im(:));
    eigen_im = eigen_im - min_x;
    scale = max_x-min_x;
    input =single (eigen_im/scale);
     %input = gpuArray(input);
    sigmas=sig/2/scale/mu;
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    BB = gather(res(end).x);
    V2(:,:,jj)=double(BB)*scale + min_x;
end
 


 V2=hyperConvert2D(V2);
G2=G2+2*mu*(V2-A);


mu=mu*para.gama;
end
Z=hyperConvert3D(D*A,nr, nc );
