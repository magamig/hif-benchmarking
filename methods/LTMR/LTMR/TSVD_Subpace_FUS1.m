function  [Z] =  TSVD_Subpace_FUS1( HSI, MSI,R,FBm,sf,S,para)
K=para.K;
eta=para.eta;

p=para.p;
mu=1e-3;
patchsize=para.patchsize;
 overlap=floor(patchsize/2);
% overlap=4;
HSI3=Unfold(HSI,size(HSI),3);
%   [D,~]=svds(HSI3,p);
% D=vca(HSI3,p);
 [D,~]=fac(HSI3);
  D=D(:,1:p);
% [w,Rn] = estNoise(HSI3);
%     [~,D]=hysime(HSI3,w,Rn);
%     D=D(:,1:p);
RD=R*D;

L1=size(D,2);




 bparams.block_sz = [patchsize, patchsize];
 bparams.overlap_sz=[overlap overlap];
[nr, nc,~]=size(MSI);
L=size(HSI,3);
bparams.sz=[nr nc];
sz=[nr nc];
% num1=ceil((nr-patchsize)/(patchsize-overlap))+1;
% num2=ceil((nc-patchsize)/(patchsize-overlap))+1;
%  bparams.block_num=[num1 num2]; 
step=patchsize-overlap;
sz1=[1:step:sz(1)- bparams.block_sz(1)+1];
 sz1=[sz1 sz1(end)+1:sz(1)- bparams.block_sz(1)+1];
sz2=[1:step:sz(2)- bparams.block_sz(2)+1];
sz2=[sz2 sz2(end)+1:sz(2)- bparams.block_sz(2)+1];
bparams.block_num(1)=length(sz1);
bparams.block_num(2)=length(sz2);

predenoised_blocks = ExtractBlocks1(MSI, bparams);
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





for i=1:50
    
     C1=DTD\(RD'*RD+mu*eye(size(D,2))); 
    
    HR_HSI3=mu*(V2+G2/(2*mu));
C3=CCC+DTD\HR_HSI3;

[A] = Sylvester(C1,FBm, sf,n_dr,n_dc,C3);   
    
Zt=hyperConvert3D(D*A,nr, nc );
% aa1=C1*ZE-C3+ creat_HSI_T(creat_HSI(ZE ,psfY),psfY);
% norm(aa1(:))

  rmse2(i)=getrmse(double((S)),double((Zt))) 

%% spatial similarties

  
B2=A-G2/(2*mu);
 B2=hyperConvert3D(B2,nr, nc );
   predenoised_blocks2 = ExtractBlocks1(B2, bparams);
   Z_block2=zeros( bparams.block_sz(1), bparams.block_sz(2),L1, bparams.block_num(1)* bparams.block_num(2));
    predenoised_blocks2=permute(predenoised_blocks2,[4 3 1 2]);
%  predenoised_blocks2=permute(predenoised_blocks2,[4 1 2 3]);
  
for mn=1:max(aa)
    gg=find(aa==mn);
 XES=predenoised_blocks2(gg,:,:,:);
   [a, b, c, d ]=size(XES);
 
    XES = reshape(XES,[a b c*d]);
    
% V22=prox_tnn( XES, eta/2/mu );
     V22=Log_prox_tnn( XES, eta/2/mu );
V22=reshape(V22,[a b c d]); 
%     Z_block2(:,:,:,gg)=permute(V22,[2 3 4 1]);
 Z_block2(:,:,:,gg)=permute(V22,[3 4 2 1]);
end
    
V2= JointBlocks2(Z_block2, bparams);
V2=V2(1:nr,1:nc,:);
V2=hyperConvert2D(V2);
G2=G2+2*mu*(V2-A);

mu=mu*1.05;
end
Z=hyperConvert3D(D*A,nr, nc );