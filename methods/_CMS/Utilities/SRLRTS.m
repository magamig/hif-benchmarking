function [PSNR_base,RMSE_base,PSNR_final,...
    RMSE_final,PSNR_pre,RMSE_pre] = SRLRTS(RZ2d,rzSize,sf,par,X,Y,H,P)
% RZ2d: input hyperspectral image(ground truth) Z
% rzSize: size of Z
% sf: scaling factor
% par: parameters
% X: low resolution HSI
% Y: high spatial resolution RGB image
% H: X = ZH
% P: Y = PZ
sz = [rzSize(1),rzSize(2)];
Z = base_bicubic(X,sf);      % baseline Bicubic interpolation
[PSNR_base,RMSE_base] = Evaluate(RZ2d, Z);

Z3d = ReshapeTo3D(Z,rzSize);
Y3d = ReshapeTo3D(Y,[rzSize(1),rzSize(2),3]);
I_1 = eye(size(P,2));

V1 = zeros(rzSize(3),rzSize(1)*rzSize(2));
V2 = zeros(rzSize(3),rzSize(1)*rzSize(2));

% XHT = par.HT(X);
XHT = X*H';

[Wi,index]= getWeightNew(Y3d,par); 

Groups = cell(1,par.nCluster);
Li = cell(1,par.nCluster);

PSNR_last = PSNR_base;
to = 1;
for t = 1 : par.iter

    uGrpPatchs = Im2Patch3D(Z3d,par);  
    sizeLi = size(uGrpPatchs);
    
    for i = 1:par.nCluster
        Groups{i} = uGrpPatchs(:,:,index{i}); 
    end

    for i = 1 : par.nCluster
        tempLi = ReshapeTo2D_C(Groups{i});        % 3-mode product
        resLi = tempLi*Wi{i};
        Li{i} = ReshapeTo3D(resLi',size(Groups{i}));
%           Li{i} = Groups{i};
    end

    Epatch          = zeros(sizeLi);
    W               = zeros(sizeLi(1),sizeLi(3));
    for i = 1:par.nCluster
        Epatch(:,:,index{i})  = Epatch(:,:,index{i}) + Li{i};
        W(:,index{i})         = W(:,index{i})+ones(size(Li{i},1),size(Li{i},3));
    end
    [L, ~]  =  Patch2Im3D( Epatch, W, par, rzSize);              % recconstruct the estimated MSI by aggregating all reconstructed FBP goups.

    L = ReshapeTo2D(L);    

    U  = ((par.mu+par.eta)^-1)*(par.mu*Z+par.eta*L+V1/2);        % Update splitting variables U with Eq.15
    S  = ((P'*P+par.mu*I_1)^-1)*(P'*Y+par.mu*Z+V2/2);            % I denotes an identity matrix with proper size.Update splitting variables U with Eq.17        

    % Solve Eq.19
    B = (XHT+par.mu*U+par.mu*S-V1/2-V2/2)';
    for j = 1: rzSize(3)
        [z,~]     =    pcg( @(x)A_x(x, par.mu, par.fft_B, par.fft_BT, sf, sz), B(:,j), 1E-3, 350, [], [], Z(j, :)' );
        Z(j, :)      =    z';
    end

    V1 = V1+2*par.mu*(Z-U);                                  % Update multipliers V1 and V2 as Eq.20;
    V2 = V2+2*par.mu*(Z-S);
    par.mu = par.rho*par.mu;
    
    [PSNR, ~] = Evaluate(RZ2d, Z);
    if PSNR - PSNR_last <0.008
        to = to + 1;
    end

    if to>3
        break;
    end
    PSNR_last = PSNR;
    
    Z3d = ReshapeTo3D(Z,rzSize);
end
[PSNR_pre, RMSE_pre] = Evaluate(RZ2d, Z);
Z = alternating_back_projection(Z,X,Y,P,H);
[PSNR_final, RMSE_final] = Evaluate(RZ2d, Z);
