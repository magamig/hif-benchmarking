function  [E_Img, Weight]  =  Patch2Im3D( ImPat, WPat, par, sizeV )
patsize      = par.patsize;
step   = par.Pstep;

TempR        =   floor((sizeV(1)-patsize)/step)+1;
TempC        =   floor((sizeV(2)-patsize)/step)+1;
TempOffsetR  =   [1:step:(TempR-1)*step+1];
TempOffsetC  =   [1:step:(TempC-1)*step+1];

E_Img  	=  zeros(sizeV);
W_Img 	=  zeros(sizeV);
k        =   0;
for i  = 1:patsize
    for j  = 1:patsize
        k    =  k+1;
        E_Img(TempOffsetR-1+i,TempOffsetC-1+j,:)  =  E_Img(TempOffsetR-1+i,TempOffsetC-1+j,:) + Fold( ImPat(k,:,:), [TempR TempC sizeV(3)], 3);
        W_Img(TempOffsetR-1+i,TempOffsetC-1+j,:)  =  W_Img(TempOffsetR-1+i,TempOffsetC-1+j,:) + Fold( repmat(WPat(k,:),sizeV(3),1), [TempR TempC sizeV(3)], 3);%reshape( WPat(k,:)',  [TempR TempC]);
    end
end
E_Img  =  E_Img./(W_Img+eps);
Weight =  1./(W_Img);

