function  W1   =  Comp_NLM_Matrix( Y, sz )
Y       =   Y*255;
ch      =   size(Y,1);
h       =   sz(1);
w       =   sz(2);
im      =   zeros(h, w, ch);
for i   =   1 : ch
    im(:,:,i)     =  reshape( Y(i,:), h, w );
end
S          =   18;
f          =   3;
f2         =   f^2; 
t          =   floor(f/2);
nv         =   10;
hp         =   90;   % 35
e_im       =   padarray( im, [t t], 'symmetric' );
[h w]      =   size( im(:,:,1) );
nt         =   (nv)*h*w;
R          =   zeros(nt,1);
C          =   zeros(nt,1);
V          =   zeros(nt,1);
L          =   h*w;
X          =   zeros(f*f*ch, L, 'single'); %前9行为R谱段周围的9个像素 中间9行为G谱段周围的9个像素 后9行为B谱段周围的9个像素（从左上角到右上角最后从左下角到右下角）
k          =   0;
for i  = 1:f
    for j  = 1:f
        
        k        =   k+1;
        blk      =   e_im(i:end-f+i,j:end-f+j, 1);
        X(k,:)   =   blk(:)'; %按列存储
        
        if ch>1
            blk  =  e_im(i:end-f+i,j:end-f+j, 2);
            X(k+f2,:) =  blk(:)';
            if ch>2
                blk  =  e_im(i:end-f+i,j:end-f+j, 3);
                X(k+f2*2,:) =  blk(:)';
            end            
        end 
     
    end
end
X           =   X'; 
X2          =   sum(X.^2, 2);
f2          =   f^2;
I           =   reshape((1:L), h, w);
f3          =   f2*ch;
cnt         =  1;
for  row  =  1 : h
    for  col  =  1 : w
        
        off_cen  =  (col-1)*h + row;        
        
        rmin    =   max( row-S, 1 );
        rmax    =   min( row+S, h );
        cmin    =   max( col-S, 1 );
        cmax    =   min( col+S, w );
         
        idx     =   I(rmin:rmax, cmin:cmax);
        idx     =   idx(:);
        B       =   X(idx, :);        
        B2      =   X2(idx, :);
        v       =   X(off_cen, :);
        v2      =   X2(off_cen, :);
        c2      =   B*v';
        
        dis     =   (B2 + v2 - 2*c2)/f3;
        [val,ind]     =   sort(dis);        
        dis(ind(1))   =   dis(ind(2));        
        wei           =   exp( -dis(ind(1:nv))./hp ); %取前nv个相似的谱线权值
        
        R(cnt:cnt+nv-1)     =   off_cen; %当前为第几个谱线
        C(cnt:cnt+nv-1)     =   idx( ind(1:nv) ); %取前nv个相似谱线
        V(cnt:cnt+nv-1)     =   wei./(sum(wei)+eps);
       
        cnt                 =   cnt + nv;        
    end
end
R     =   R(1:cnt-1); %与原位置谱线相似的nv个谱线的横坐标
C     =   C(1:cnt-1); %与原位置谱线相似的nv个谱线的纵坐标
V     =   V(1:cnt-1); %与原位置谱线相似的nv个谱线的权值
% W1    =   spdiags(ones(h*w,1), 0, h*w, h*w)-sparse(R, C, V, h*w, h*w); %相似谱线的权值矩阵
W1    =   sparse(R, C, V, h*w, h*w); %相似谱线的权值矩阵
W1    =   W1';
