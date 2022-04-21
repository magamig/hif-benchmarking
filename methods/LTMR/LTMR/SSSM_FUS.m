function  [Z] =  SSSM_FUS( HSI, MSI,R, K,FBm,sf,S)
mu=1e-3;
eta=1e-3;
eta1=1e-3;
eta2=1e-2;
patchsize=8;
overlap=4;
 bparams.block_sz = [patchsize, patchsize];
 bparams.overlap_sz=[overlap overlap];
[nr, nc,~]=size(MSI);
L=size(HSI,3);
num1=(nr-patchsize)/(patchsize-overlap)+1;
num2=(nc-patchsize)/(patchsize-overlap)+1;
 bparams.block_num=[num1 num2]; 

fkmeans_omaxpt.careful = 1;
predenoised_blocks = ExtractBlocks(MSI, bparams);
Y2=Unfold(predenoised_blocks,size(predenoised_blocks),4);
  [aa ]=fkmeans(Y2,K,fkmeans_omaxpt);


 HSI_int=zeros(nr,nc,L);
    HSI_int(1:sf:end,1:sf:end,:)=HSI;
 
    FBmC  = conj(FBm);
    FBs  = repmat(FBm,[1 1 L]);
    
FBCs1=repmat(FBmC,[1 1 L]);
HHH=ifft2((fft2(HSI_int).*FBCs1));
  HHH1=hyperConvert2D(HHH);
psfY.w=nr/sf;
psfY.h=nc/sf;
psfY.W=nr;
psfY.H=nc;

Zt=HSI_int;
HR_load1=imresize(HSI, sf,'bicubic');

ZE=hyperConvert2D(HR_load1);

%% iteration

MSI3=Unfold(MSI,size(MSI),3);

n_dr=nr/sf;
n_dc=nc/sf;

V1=ZE;
V2=ZE;

G1=zeros(size(V1));
G2=zeros(size(V2));

CCC=R'*MSI3+HHH1;
 C1=R'*R+2*mu*eye(size(R,2)); 
 [Q,Lambda]=eig(C1);
Lambda=reshape(diag(Lambda),[1 1 L]);
InvLbd=1./repmat(Lambda,[ sf*n_dr  sf*n_dc 1]);
B2Sum=PPlus(abs(FBs).^2./( sf^2),n_dr,n_dc);
InvDI=1./(B2Sum(1:n_dr,1:n_dc,:)+repmat(Lambda,[n_dr n_dc 1]));
HSI3=Unfold(HSI,size(HSI),3);
 [SSS, b1]=learn_spectral_similarties(HSI3,eta);
%  SSS1=eye(size(SSS,2))-SSS;
%  SSS2=SSS1'*SSS1;
for i=1:12



    HR_HSI3=mu*(V1+G1/(2*mu)+V2+G2/(2*mu));
   

  

C3=CCC+HR_HSI3;





C30=fft2(reshape((Q\C3)',[nr nc L   ])).*InvLbd;

temp  = PPlus_s(C30/( sf^2).*FBs,n_dr,n_dc);
invQUF = C30-repmat(temp.*InvDI,[ sf  sf 1]).*FBCs1; % The operation: C5bar- temp*(\lambda_j d Im+\Sum_i=1^d Di^2)^{-1}Dv^H)
VXF    = Q*reshape(invQUF,[nc*nc L])';
ZE = reshape(real(ifft2(reshape(VXF',[nr nc L   ]))),[nc*nc L])'; 



%   [ZE1] = Sylvester(C1,psfY.B, sf,n_dr,n_dc,C3);  
        
 
Zt=hyperConvert3D(ZE,nr, nc );
% aa1=C1*ZE-C3+ creat_HSI_T(creat_HSI(ZE ,psfY),psfY);
% norm(aa1(:))

  rmse2(i)=getrmse(double((S)),double((Zt))) 
 
%% spectral  similarties

  B1=ZE-G1/(2*mu);
  
  for i=1:length(b1)-1
    A=B1(b1(i)+1:b1(i+1),:);
    SSS1=eye(size(SSS{i},2))-SSS{i};
    SSS2=SSS1'*SSS1;
      V1(b1(i)+1:b1(i+1),:)=(eta1*SSS2+mu*eye(size(SSS2,2)))\(mu*A);
end
%   V1=(eta1*SSS2+mu*eye(size(SSS2,2)))\(mu*B1);
%   V1=SSS*ZE;
  
%% spatial similarties

  
B2=ZE-G2/(2*mu);
 B2=hyperConvert3D(B2,nr, nc );
   predenoised_blocks2 = ExtractBlocks(B2, bparams);
  Z_block2=zeros( bparams.block_sz(1), bparams.block_sz(2),L, bparams.block_num(1)* bparams.block_num(2));
  
  
  
for mn=1:max(aa)
    gg=find(aa==mn);
  


  nsig2=1;

 XES=predenoised_blocks2(:,:,:,gg);
   [a, b, c, d ]=size(XES);
   XES = reshape(XES,[a*b*c d]);
 V22=WNNM( XES, eta2/2/mu, nsig2 );
V22=reshape(V22,[a b c d]);
    Z_block2(:,:,:,gg)=V22;
    
    
    
end
    
V2= JointBlocks(Z_block2, bparams);
V2=hyperConvert2D(V2);
% tt1(i)=(norm(2*mu*(V1-ZE),'fro')+norm(2*mu*(V2-ZE),'fro'))/(norm(G1,'fro')+norm(G2,'fro'))
% if tt1<0.15
%     break
% end
G1=G1+2*mu*(V1-ZE);
 G2=G2+2*mu*(V2-ZE);

end
Z=Zt;