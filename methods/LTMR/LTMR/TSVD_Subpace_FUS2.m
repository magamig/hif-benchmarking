function  [Z] =  TSVD_Subpace_FUS2( HSI, MSI,R,FBm,sf,S,para)
K=para.K;
eta=para.eta;

p=para.p;
mu=1e-3;
patchsize=8;
overlap=4;

HSI3=Unfold(HSI,size(HSI),3);
[D,~]=svds(HSI3,p);
RD=R*D;
L1=size(D,2);




 bparams.block_sz = [patchsize, patchsize];
 bparams.overlap_sz=[overlap overlap];
[nr, nc,~]=size(MSI);
L=size(HSI,3);
num1=(nr-patchsize)/(patchsize-overlap)+1;
num2=(nc-patchsize)/(patchsize-overlap)+1;
 bparams.block_num=[num1 num2]; 

predenoised_blocks = ExtractBlocks(MSI, bparams);
Y2=Unfold(predenoised_blocks,size(predenoised_blocks),4);
if K==1
    aa=ones(num1*num2,1);
else
  [aa ]=fkmeans(Y2,K);
end

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

V2=D'*hyperConvert2D(HR_load1);



G2=zeros(size(V2));
DTD=D'*D;
CCC=DTD\(RD'*MSI3+D'*HHH1);
 C1=DTD\(RD'*RD+mu*eye(size(D,2))); 
 [Q,Lambda]=eig(C1);
Lambda=reshape(diag(Lambda),[1 1 L1]);
InvLbd=1./repmat(Lambda,[ sf*n_dr  sf*n_dc 1]);
B2Sum=PPlus(abs(FBs1).^2./( sf^2),n_dr,n_dc);
InvDI=1./(B2Sum(1:n_dr,1:n_dc,:)+repmat(Lambda,[n_dr n_dc 1]));



for i=1:100
    HR_HSI3=mu*(V2+G2/(2*mu));
C3=CCC+DTD\HR_HSI3;
C30=fft2(reshape((Q\C3)',[nr nc L1   ])).*InvLbd;
temp  = PPlus_s(C30/( sf^2).*FBs1,n_dr,n_dc);
invQUF = C30-repmat(temp.*InvDI,[ sf  sf 1]).*FBCs1; % The operation: C5bar- temp*(\lambda_j d Im+\Sum_i=1^d Di^2)^{-1}Dv^H)
VXF    = Q*reshape(invQUF,[nc*nc L1])';
A = reshape(real(ifft2(reshape(VXF',[nr nc L1   ]))),[nc*nc L1])'; 
%   [ZE1] = Sylvester(C1,psfY.B, sf,n_dr,n_dc,C3);  
    
Zt=hyperConvert3D(D*A,nr, nc );
% aa1=C1*ZE-C3+ creat_HSI_T(creat_HSI(ZE ,psfY),psfY);
% norm(aa1(:))

  rmse2(i)=getrmse(double((S)),double((Zt))) 

%% spatial similarties

  
B2=A-G2/(2*mu);
 B2=hyperConvert3D(B2,nr, nc );
   predenoised_blocks2 = ExtractBlocks(B2, bparams);
   Z_block2=zeros( bparams.block_sz(1), bparams.block_sz(2),L1, bparams.block_num(1)* bparams.block_num(2));
%     predenoised_blocks2=permute(predenoised_blocks2,[4 3 1 2]);
%  predenoised_blocks2=permute(predenoised_blocks2,[4 1 2 3]);
  
for mn=1:max(aa)
    gg=find(aa==mn);
 XES=predenoised_blocks2(:,:,:,gg);
   [a, b, c, d ]=size(XES);
 
    XES = reshape(XES,[a*b c d]);
    
% V22=prox_tnn( XES, eta/2/mu );
     V22=Log_prox_tnn( XES, eta/2/mu );
V22=reshape(V22,[a b c d]); 
%     Z_block2(:,:,:,gg)=permute(V22,[2 3 4 1]);
  Z_block2(:,:,:,gg)=V22;
end
    
V2= JointBlocks(Z_block2, bparams);
V2=hyperConvert2D(V2);
G2=G2+2*mu*(V2-A);


end
Z=hyperConvert3D(D*A,nr, nc );