function [  ] = BayesianHSISupRes( param )
%--------------------------------------------------------------
% Set the default values of the parametes
%--------------------------------------------------------------
HSI = 'img2';        % Image to be tested (provided)
a0 = 1e-6;               
b0 = 1e-6;               
c0 = 1e-6;
d0 = 1e-6;
e0 = 1e-6;
f0 = 1e-6;
DLiterations = 50000;  
SCiterations = 100;    % Number of iterations per single run of the sparse coding stage
Q = 32; 

%--------------------------------------------------------------
% Read the parameters, if given
%--------------------------------------------------------------
if isfield(param, 'HSI')
    HSI = param.HSI;
end
param.HSI = HSI;

if isfield(param, 'a0')
    a0 = param.a0;
end
param.a0 = a0;

if isfield(param, 'b0')
    b0 = param.b0;
end
param.b0 = b0;

if isfield(param, 'c0')
    c0 = param.c0;
end
param.c0 = c0;

if isfield(param, 'd0')
    d0 = param.d0;
end
param.d0 = d0;

if isfield(param, 'e0')
    e0 = param.e0;
end
param.e0 = e0;

if isfield(param, 'f0')
    f0 = param.f0;
end
param.f0 = f0;

if isfield(param, 'DLiterations')
    DLiterations = param.DLiterations;
end
param.DLiterations = DLiterations;

if isfield(param, 'SCiterations')
    SCiterations = param.SCiterations;
end
param.SCiterations = SCiterations;

if isfield(param, 'Q')
    Q = param.Q;
end
param.Q = Q;

%--------------------------------------------------------------
% Simulated Nikon D700 spectral response T = [R;G;B], 
% Please note, the same transformation must be used between 'the ground
% truth and RGB image' and the leanred and the transformed dictionary.
%--------------------------------------------------------------
T1 = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004   0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0];
T2 = [0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003 ];
T3 = [0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];
T  = [T1;T2;T3];

%--------------------------------------------------------------
% Read the input image and downsample
%--------------------------------------------------------------
HSI = param.HSI;
downsampling_scale = 32;
im_structure = load(HSI);
S = im_structure.im;
S = S(1:128,1:128,:);
[M,N,L] = size(S);
% If the image is not square, make it so, to match the downspamling
min_dim = min(M,N);
dim = floor(min_dim/8)*8;
S = S(1:dim, 1:dim, :);
[Y_h] = downsample(S, downsampling_scale);
Y_h_bar = hyperConvert2D(Y_h);

%--------------------------------------------------------------
% Simulate the high resolution image (i.e. RGB image)
%--------------------------------------------------------------
S_bar = hyperConvert2D(S);
Y = hyperConvert3D((T*S_bar), M, N, 3);
Y_bar = hyperConvert2D(Y);


%--------------------------------------------------------------
% Learn Dictionary
%--------------------------------------------------------------
[Phi,Pi,g_eps,g_s] = BayesianDictionaryLearning(Y_h_bar,param);
Phi_cap = T*Phi;

%--------------------------------------------------------------
% Sparse code
%--------------------------------------------------------------
param.Pi = Pi;
param.g_s = g_s;
param.g_eps = g_eps;
Acum = zeros(size(Phi_cap,2),size(Y_bar,2));
Q = param.Q;
disp(' ')
disp('Bayesian sparse coding....')
disp('Progress: 0% ...')
tic
for i = 1:Q
    t = 100 * (i/Q);
    [A] = simpleBayesianSC(Y_bar,Phi_cap, param);
    Acum = Acum + A;
    disp(['Progress: ' num2str(t) '% ...'])
end
A_tilde = (1/Q)*(Acum);
T_bar = Phi*A_tilde;
timer  = toc;
disp(['Total time: ' num2str(timer) ' seconds'])
disp(['Time per single run: ' num2str(timer/Q) ' seconds'])
T = hyperConvert3D(T_bar,M,N,L);
beep
S_bar = im2uint8(S_bar);
T_bar = im2uint8(T_bar);
RMSE = sqrt((norm((double(S_bar) - double(T_bar)),2)).^2/(M*N*L))
%--------------------------------------------------------------
% Display images
%--------------------------------------------------------------
r = 7; g = 15; b = 21;
err460 = abs((double(im2uint8(S(:,:,r)))- double(im2uint8(T(:,:,r)))));
err540 = abs((double(im2uint8(S(:,:,g)))- double(im2uint8(T(:,:,g)))));
err620 = abs((double(im2uint8(S(:,:,b)))- double(im2uint8(T(:,:,b)))));

figure(1),subplot(3,3,1),imshow(Y_h(:,:,r)),title('460nm (Input)'),subplot(3,3,2),imshow(Y_h(:,:,g)),title('540nm (Input)'),subplot(3,3,3),imshow(Y_h(:,:,b)),title('620nm (Input)'), subplot(3,3,4),imshow(S(:,:,r)),title('460nm (Ground truth)'),subplot(3,3,5),imshow(S(:,:,g)),title('540nm (Ground truth)'),subplot(3,3,6),imshow(S(:,:,b)),title('620nm (Ground truth)'), subplot(3,3,7),imshow(T(:,:,r)),title('460nm (Estimate)'),subplot(3,3,8),imshow(T(:,:,g)),title('540nm (Estimate)'),subplot(3,3,9),imshow(T(:,:,b)),title('620nm (Estimate)')   
figure(2),subplot(1,3,1),imagesc(err460, [0 25]),colorbar, title('460nm (Absolute error)'), subplot(1,3,2), imagesc(err540, [0 25]),colorbar,title('540nm (Absolute error)'),subplot(1,3,3), imagesc(err620, [0 25]),colorbar,title('620nm (Absolute error)')
end

function [ D, Pi,g_eps,g_s] = BayesianDictionaryLearning(X,pars) 
%Some parts of the code are taken directly form the implementation provided with [20].
    X_k = X;
    [L,mn] = size(X_k);
    K = 50;
    pars.K = K;
    Kinit = K;
    a0= pars.a0;
    b0= pars.b0;
    c0= pars.c0;
    d0= pars.d0;
    e0= pars.e0; 
    f0= pars.f0;
    
    % init D
    [D,S,g_eps,g_s] = Dictionary_Init(X_k,pars);  
    % init Z
    Z = logical(sparse(mn,K));
    % init S
    S = S.*Z;
    % init Pi
    Pi = 0.005*ones(1,K);  %Smaller values to help faster convergence of the demo.
    
    X_k = X_k - D*S';
    a = zeros(size(S));
    Z_avg=[];
    D_avg=zeros(L,K);
    K_all=[];
    time=0;
    avg_count=0;
    disp('Learning dictionary using Gibbs sampling....');
    tStart=tic;
    for iter=1:pars.DLiterations
        istart=tic;
        [X_k, D, S, Z] = my_sample_DZS(X_k,D, S, Z, Pi, g_s, g_eps);
        Pi = my_sample_Pi(Z,a0,b0);
        g_s = my_sample_g_s(S,c0,d0,Z,g_s);
        g_eps = my_sample_g_eps(X_k,e0,f0);
        ittime=toc(istart);
        nstd(iter) = sqrt(1/g_eps);
        time=time+ittime;        
        %Reduce atoms
        Z = full(Z);
        sumZ = sum(Z)';
        if min(sumZ)==0
            Pidex = sumZ==0;
            D(:,Pidex)=[];
            D_avg(:,Pidex)=[];
            K = size(D,2);
            Z(:,Pidex)=[];Pi(Pidex)=[];
            S(:,Pidex)=[];
            if (~isempty(a)) 
                a(:,Pidex)=[]; 
            end
        end     
        if (iter>iter-100)
            D_avg=D_avg+D;
            a = a+S.*Z;
            avg_count=avg_count+1;
        end 
    end    
    disp(['Initial K: ' num2str(Kinit) ' atoms']);
    disp(['Computed K: ', num2str(K) ' atoms']);
    D_avg=D_avg/avg_count;
    D=D_avg; 
    train_time=toc(tStart);  
    disp(['Total time: ' num2str(train_time) ' seconds'] );
end

function g_s = my_sample_g_s(S,c0,d0,Z,g_s)
c = c0 + 0.5*numel(Z);   
d = d0 + 0.5*sum(sum(S.^2)) + 0.5*(numel(Z)-nnz(Z))*(1/g_s);
g_s = gamrnd(c,1./d);
end

function g_eps = my_sample_g_eps(X_k,e0,f0)
e = e0 + 0.5*numel(X_k);
f = f0 + 0.5*sum(sum((X_k).^2));
g_eps = gamrnd(e,1./f);
end

function Pi = my_sample_Pi(Z, a0,b0)
sumZ = full(sum(Z));
[N,K] = size(Z);
Pi = betarnd(sumZ+(a0/K), (b0*(K-1)/K)+N-sumZ);
end

function [X_k, D, S, Z] = my_sample_DZS(X_k,D, S, Z, Pi, g_s, g_eps)
[P,N] = size(X_k);
K = size(D,2);
g_s = repmat(g_s,1,K);
for k = 1:K   
    nnzk = nnz(Z(:,k));    
    if nnzk>0
        X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + D(:,k)*S(Z(:,k),k)';           
    end
    sig_Dk = 1./(g_eps*sum(S(Z(:,k),k).^2) + P);
    mu_Dk = g_eps*sig_Dk* (X_k(:,Z(:,k))*S(Z(:,k),k));
    D(:,k) = mu_Dk + randn(P,1)*sqrt(sig_Dk);      
    DTD = sum(D(:,k).^2);   
    Mu_D(:,k) = mu_Dk;         
    Sk = full(S(:,k));
    Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/g_s(k));
    temp = exp(-0.5*g_eps*( (Sk.^2 )*DTD - 2*Sk.*(X_k'*D(:,k)))).*Pi(:,k);
    Z(:,k) = sparse( rand(N,1) > ((1-Pi(:,k))./(temp+1-Pi(:,k))));
    nnzk = nnz(Z(:,k)); 
    if nnzk>0
    sigS1 = 1/(g_s(k) + g_eps*DTD);
    S(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(g_eps*(X_k(:,Z(:,k))'*D(:,k))),N,1);                              
    end 
    if nnzk>0
        X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
    end
end
end

function [D,S,g_eps,g_s] = Dictionary_Init(X_k,pars)
% Dictionary initialization
K = pars.K;
[L,mn] = size(X_k);
g_eps = 1/(var(X_k(:))); 
g_s = 1;
% random initialization
D = rand(L,K)-0.5;
D = D - repmat(mean(D,1), size(D,1),1);
D = D*diag(1./sqrt(sum(D.*D)));
S = randn(mn,K);
end

function [down_im] = downsample(im, scale)
new_pixel_size = scale;
a = size(im,1)/new_pixel_size;
down_im = zeros(a,a,size(im,3));
s = 1:new_pixel_size:size(im,1);
e = new_pixel_size:new_pixel_size:size(im,1);

for l = 1:size(im,3)
    ima(:,:) = im(:,:,l);
    temp1 = [];
    for i = 1:a
        t = sum(ima(s(i):e(i), :));
        temp1 = [temp1; t];
    end 
    temp2 = [];
    for i = 1:a 
        t = sum(temp1(:,s(i):e(i)),2);
        temp2 = [temp2 t];
    end
    temp2 = temp2./repmat(new_pixel_size^2,size(temp2,1),size(temp2,1));
    down_im(:,:,l) = temp2;
end 
end

function [Image2D] = hyperConvert2D(Image3D)
if (ndims(Image3D) == 2)
    numBands = 1;
    [h, w] = size(Image3D);
else
    [h, w, numBands] = size(Image3D);
end
Image2D = reshape(Image3D, w*h, numBands).';
end

function [ A ] = simpleBayesianSC(X, D, pars) 
    X_k = X;
    K = size(D,2);
    % Set Hyperparameters
    c0=pars.c0;
    d0=pars.d0;
    e0=pars.e0; 
    f0=pars.f0;   
    Pi = pars.Pi; 
    g_eps = pars.g_eps;
    g_s = pars.g_s*35; %scalar multiplied for faster convergence of the demo  

    %%%%%%%%%%%%%%%%%Initialization%%%%%%%%%%%%%
    [P,N] = size(X_k);
    S = randn(N,K);
    Z = logical(sparse(N,K));
    S = S.*Z;
    X_k = X_k - D*S';
    for iter=1:pars.SCiterations  
        [X_k, S, Z] = my_sample_ZS(X_k,D, S, Z, Pi, g_s, g_eps);
        g_s = my_sample_g_s(S,c0,d0,Z,g_s);
        g_eps = my_sample_g_eps(X_k,e0,f0);
    end    
    A =(S.*Z)';
end

function [X_k, S, Z] = my_sample_ZS(X_k, D, S, Z, Pi, g_s, g_eps)
[P,N] = size(X_k);
K = size(D,2);
g_s = repmat(g_s,1,K);

for k = 1:K   
    nnzk = nnz(Z(:,k));    
    if nnzk>0
        X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + D(:,k)*S(Z(:,k),k)';           
    end   
    DTD = sum(D(:,k).^2);          
    Sk = full(S(:,k));
    Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/g_s(k));
    temp = exp(-0.5*g_eps*( (Sk.^2 )*DTD - 2*Sk.*(X_k'*D(:,k)))).*Pi(:,k);
    Z(:,k) = sparse( rand(N,1) > ((1-Pi(:,k))./(temp+1-Pi(:,k))));
    
    nnzk = nnz(Z(:,k)); 
    if nnzk>0
    sigS1 = 1/(g_s(k) + g_eps*DTD);
    S(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(g_eps*(X_k(:,Z(:,k))'*D(:,k))),N,1);                            
    end
   
    if nnzk>0
        X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
    end
 
end

end

function [img] = hyperConvert3D(img, h, w, numBands)
[numBands, N] = size(img);
if (1 == N)
    img = reshape(img, h, w);
else
    img = reshape(img.', h, w, numBands); 
end
end