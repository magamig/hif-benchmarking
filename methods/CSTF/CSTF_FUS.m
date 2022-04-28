function HR_HSI= CSTF_FUS(HSI,MSI,T,BW,BH,downsampling_scale,par,s0,S)
mu=0.001;

 lambda=par.lambda;

    


%%  simulate LR-HSI



 
 
  Y_h_bar=hyperConvert2D(HSI);
HSI1=Unfold(HSI,size(HSI),1);
HSI2=Unfold(HSI,size(HSI),2);
HSI3=Unfold(HSI,size(HSI),3);




  %%  simulate HR-MSI



MSI1=Unfold(MSI,size(MSI),1);
MSI2=Unfold(MSI,size(MSI),2);
MSI3=Unfold(MSI,size(MSI),3);



%% inilization D1 D2 D3 C
 [m n]=size(MSI1);
    D=MSI1;
     params.Tdata = 2;            % Number of target sparse vectors
      params.dictsize =par.W;      %   441;      % Number of dictionary elements (# columns of D)
      params.iternum = 100;
       params.DUC =1; 
      
       D1 = trainD(D,MSI1,[],[],params);
       
        params.dictsize =par.H;
        D2 = trainD(MSI2,MSI2,[],[],params);
      
%        D3= sisal(Y_h_bar,par.S, 'spherize', 'no','MM_ITERS',80, 'TAU', 0.006, 'verbose',0);
D3=vca(Y_h_bar,par.S);



  D_1=ifft(fft(D1).*repmat(BW,[1 par.W]));
  D_1=D_1(s0:downsampling_scale:end,:);
  
   D_2=ifft(fft(D2).*repmat(BH,[1 par.H]));
  D_2=D_2(s0:downsampling_scale:end,:);
  D_3=T*D3;
  
    D11{1}=D_1;
  D11{2}=D_2;
   D11{3}=D3;
  D22{1}=D1;
   D22{2}=D2;
  D22{3}=D_3;
  C=zeros(size(D1,2),size(D2,2),size(D3,2));
   C   =  sparse_tucker2( D11,D22, HSI,MSI, lambda,C,mu );
   
   

for i=1:15

  %%  update D1
 CC=ttm(tensor(C),{D2,D_3},[2 3]);
CC2=Unfold(double(CC),size(CC),1);
  CC3=ttm(tensor(C),{D_2,D3},[2 3]);
CC4=Unfold(double(CC3),size(CC3),1);
  [ D1 ] = DIC_CG1( MSI1, D1, CC2,HSI1,downsampling_scale,s0,BW,CC4,mu);
%     D_1=sample*D1;
 D_1=ifft(fft(D1).*repmat(BW,[1 par.W]));
  D_1=D_1(s0:downsampling_scale:end,:);
  
    %%  update D2
 CC=ttm(tensor(C),{D1,D_3},[1 3]);
CC2=Unfold(double(CC),size(CC),2);
  CC3=ttm(tensor(C),{D_1,D3},[1 3]);
CC4=Unfold(double(CC3),size(CC3),2);
  [ D2 ] = DIC_CG1( MSI2, D2, CC2,HSI2,downsampling_scale,s0,BH,CC4,mu );
    D_2=ifft(fft(D2).*repmat(BH,[1 par.H]));
  D_2=D_2(s0:downsampling_scale:end,:);

   %%  update D3
  CC=ttm(tensor(C),{D_1,D_2},[1 2]);
CC2=Unfold(double(CC),size(CC),3);
CC3=ttm(tensor(C),{D1,D2},[1 2]);
CC4=Unfold(double(CC3),size(CC3),3);
  [ D3 ] = DIC_CG( HSI3, D3, CC2,MSI3,T,CC4,mu );
  D_3=T*D3;
  
 %%  update C
    D11{1}=D_1;
  D11{2}=D_2;
   D11{3}=D3;
  D22{1}=D1;
   D22{2}=D2;
  D22{3}=D_3;
   C   =  sparse_tucker2( D11,D22, HSI,MSI, lambda,C,mu );
  
  

    
 HR_HSI=ttm(tensor(C),{D1,D2,D3},[1 2 3]);
 HR_HSI=double(HR_HSI);
%     HR_HSI(HR_HSI<0)=0;
% HR_HSI(HR_HSI>1)=1;
% Z1=hyperConvert2D(HR_HSI);
% b11=alternating_back_projection(Z1,X,Y,T,G);
% b12=hyperConvert3D(b11,M,N,L);
% 
%rmse1(i)=getrmse(double(im2uint8(S)),double(im2uint8(HR_HSI)))
%  rmse2(i)=getrmse(double(im2uint8(S)),double(im2uint8(b12)))

end

       
HR_HSI=ttm(tensor(C),{D1,D2,D3},[1 2 3]);

 HR_HSI=double(HR_HSI);


