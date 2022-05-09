function [x] = deconvs_auto(y,y_hat,mu,nu)
% DECONVS deconvolution of spectral images
% x = deconvs_3regul(y,h,mu,nu,conv,regu,pos)
% y  : N1 x N2 x L observed spectral data
% mu : spatial regularization parameter
% nu : spectral regularization parameter
% conv : convolution type
%      * 'direct' assumes the original convolution is direct, i.e.
%      y = conv2(x0,h,'full') where x0 is (N1-P1+1)x(N2-P2+1)
%      * 'circular' assumes the original convolution is circular, i.e.
%      y = real(ifft2(fft2(x0).*fft2(h))) where x0 is N1xN2
% regu : spatial regularization type : 'l2', 'l2l1_sq', 'l2l1_admm', 'l1'
% pos  : positivity constraint : 'on', 'off'
% last update : 09/06/13

%% main

% data size
N1 = size(y,1); N2 = size(y,2); L = size(y,3);

% spatial regularization operator
laplacian = [0,-1,0;-1,4,-1;0,-1,0];
d = reshape(repmat(laplacian,1,L),3,3,L);

% spectral regularization operator
T0 = toeplitz([-1;zeros(L-2,1)],[-1 1 zeros(1,L-2)]);
E0 = sparse([T0;zeros(1,size(T0,2))]);


% fourier domain
Y = fft2(y,N1,N2);
Y_hat = fft2(y_hat,N1,N2);
D = fft2(d,N1,N2);

% gather variables in struct variables to call other functions
variables = struct('data',Y,'data_hat',Y_hat,'spatial_operator',D,...
    'spectral_operator',E0,'spatial_param',mu,...
    'spectral_param',nu);

X = sym(variables);
xf = real(ifft2(X));

% output image
x = xf;

end

%% compute spectrum at frequencies f1 and f2
function val = comp(variables,f1,f2)

Y = variables.data;
Y_hat = variables.data_hat;
L = size(Y,3);
D = variables.spatial_operator;
E0 = variables.spectral_operator;
mu = variables.spatial_param;
nu = variables.spectral_param;

% computes spectrum at f1 and f2
vy = squeeze(Y(f1,f2,:));
vy_hat = squeeze(Y_hat(f1, f2,:));
delta_D = diag(squeeze(D(f1, f2,:)));

A = mu*(delta_D'*delta_D) + eye(L) + nu*(E0'*E0);
bv = vy + mu*(delta_D'*delta_D) * vy_hat + nu*(E0'*E0) * vy_hat;

val = A\bv;
end

%% compute full spectrum with hermitian symmetry
function spec = sym(variables)

Y = variables.data;
N1 = size(Y,1); N2 = size(Y,2); L = size(Y,3);
spec = zeros(N1,N2,L);

if rem(N1,2)
    r1 = (N1+1)/2; r2 = r1;
else
    r1 = N1/2+1; r2 = r1-1;
end
if rem(N2,2)
    c1 = (N2+1)/2; c2 = c1;
else
    c1 = N2/2+1; c2 = c1-1;
end

% compute 1st row, columns 1 : c2
for col=1:c1
    spec(1,col,:) = comp(variables,1,col);
end
% compute rows 2 : ind
for row=2:r1
    for col=1:N2
        spec(row,col,:) = ...
            comp(variables,row,col);
    end
end
% fill in spectrum
for l=1:L
    % 1st row, columns c+1 : N
    spec(1,c1+1:N2,l) = fliplr(conj(spec(1,2:c2,l)));
    % 1st column, rows k+1 : N
    spec(r1+1:N1,1,l) = flipud(conj(spec(2:r2,1,l)));
    % rows ind+1 : M, columns 2 : N
    spec(r1+1:N1,2:N2,l) = rot90(conj(spec(2:r2,2:N2,l)),2);
end
end
