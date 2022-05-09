function [Z] = Sylvester(C1,FBm,ds_r,n_dr,n_dc,C3)
L=size(C1,2);
nr=ds_r*n_dr;
nc=ds_r*n_dc;


% FBm   = fft2(B); 
    FBmC  = conj(FBm);
    FBs  = repmat(FBm,[1 1 L]);
  
FBCs1=repmat(FBmC,[1 1 L]);


[Q,Lambda]=eig(C1);
Lambda=reshape(diag(Lambda),[1 1 L]);
InvLbd=1./repmat(Lambda,[ds_r*n_dr ds_r*n_dc 1]);
B2Sum=PPlus(abs(FBs).^2./(ds_r^2),n_dr,n_dc);
InvDI=1./(B2Sum(1:n_dr,1:n_dc,:)+repmat(Lambda,[n_dr n_dc 1]));
C30=fft2(reshape((Q\C3)',[nr nc L   ])).*InvLbd;

temp  = PPlus_s(C30/(ds_r^2).*FBs,n_dr,n_dc); % The operation: temp=1/d*C5bar*Dv
 

invQUF = C30-repmat(temp.*InvDI,[ds_r ds_r 1]).*FBCs1; % The operation: C5bar- temp*(\lambda_j d Im+\Sum_i=1^d Di^2)^{-1}Dv^H)
VXF    = Q*reshape(invQUF,[nc*nc L])';
Z = reshape(real(ifft2(reshape(VXF',[nr nc L   ]))),[nc*nc L])'; 
