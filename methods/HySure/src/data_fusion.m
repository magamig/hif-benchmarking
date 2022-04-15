function [ Zimhat ] = data_fusion( Yhim, Ymim, downsamp_factor, R, B, p, basis_type, lambda_phi, lambda_m )
%data_fusion - The actual data fusion algorithm
% 
%   Given the two observed images (HS + PAN/MS), fuses the data according
%   to the steps described in detail in the Appendix of [1]. 
% 
% [ Zimhat ] = data_fusion( Yhim, Ymim, downsamp_factor, R, B, p, 
%                             basis_type, lambda_phi, lambda_m )
% 
% Input: 
% Yhim: HS image, 
% Ymim: MS image, 
% downsamp_factor: downsampling factor,
% R: relative spectral resppnse;
% B: relative spatial response;
% p: corresponds to variable L_s in [1]; number of endmembers in VCA /
%     number of non-truncated singular vectors,
% basis_type: method to estimate the subspace. Can be either 'VCA' or
%     'SVD'
% lambda_phi: regularization parameter lambda_phi (corresponds to VTV), 
% lambda_m: regularization parameter lambda_m.
% 
% Output: 
% Zimhat: estimated image with high spatial and spectral resolution
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A convex formulation for hyperspectral image superresolution 
%        via subspace-based regularization,” IEEE Trans. Geosci. Remote 
%        Sens., to be publised.

% % % % % % % % % % % % % 
% 
% Version: 1
% 
% Can be obtained online from: https://github.com/alfaiate/HySure
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2014 Miguel Simoes, Jose Bioucas-Dias, Luis B. Almeida 
% and Jocelyn Chanussot
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% 
% % % % % % % % % % % % % 
% 
% This funtion has three steps:
% 
% I. Precomputations (for example, FFTs of the filters that will be used 
% throughout the function). 
% 
% II. It will then learn the subspace where the HS ''lives'', via SVD
% or VCA.
% 
% Basis type - 1: find endmembers with VCA, 3: learn the
%     subspace with SVD
% 
% III. The iterative process used to compute a solution to the
% optimization problem via ADMM/SALSA. The following parameters can be 
% adjusted:
% ADMM parameter
mu = 0.05;
% ADMM iterations
iters = 200;
% 
% IV. Postprocessing (denoising).
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% I. Precomputations.                                                   %
% -------------------                                                   %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

[nl, nc, ~] = size(Ymim);
% Define the difference operators as filters
dh = zeros(nl,nc);
dh(1,1) = 1;
dh(1,nc) = -1;

dv = zeros(nl,nc);
dv(1,1) = 1;
dv(nl,1) = -1;

FDH = fft2(dh);
FDHC = conj(FDH);
FDV = fft2(dv);
FDVC = conj(FDV);

% Fourier transform of B
FB = fft2(B);
FBC = conj(FB);

IBD_B  = FBC  ./(abs(FB.^2) + abs(FDH).^2+ abs(FDV).^2 + 1);
IBD_II = 1    ./(abs(FB.^2) + abs(FDH).^2+ abs(FDV).^2 + 1);
IBD_DH = FDHC ./(abs(FB.^2) + abs(FDH).^2+ abs(FDV).^2 + 1);
IBD_DV = FDVC ./(abs(FB.^2) + abs(FDH).^2+ abs(FDV).^2 + 1);

% % % % % % % % % % % %
% We will work with image Yhim in a matrix with the same size as Ymim. The
% result is a matrix filled with zeros. We do this for computational 
% convenience and the end result is the same. We also explicity form a
% subsampling mask that has the same effect has matrix M in [1].
shift = 1;
mask = zeros(nl, nc);
mask(ceil(downsamp_factor/2)+shift-1:downsamp_factor:nl+shift-1, ceil(downsamp_factor/2)+shift-1:downsamp_factor:nc+shift-1) = 1;
% Subsampling mask in image format
maskim = repmat(mask, [1, 1, p]);
% Subsampling mask in matrix format
mask = im2mat(maskim);
% Yhim with the same size as Ym (but with zeros)
Yhim_up = upsamp_HS(Yhim, downsamp_factor, nl, nc, shift);
Yh_up = im2mat(Yhim_up);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% II. Subspace learning.                                                %
% ----------------------                                                %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
switch  basis_type 
    case 'VCA'
%     Find endmembers with VCA (pick the one with smallest volume from 20 
%     runs of the algorithm)
    max_vol = 0;
    vol = zeros(1, 20);
    for idx_VCA = 1:20
        E_aux = VCA(Yh_up(:, mask(1,:) > 0),'Endmembers',p,'SNR',0,'verbose','off');
        vol(idx_VCA) = abs(det(E_aux'*E_aux));
        if vol(idx_VCA) > max_vol
            E = E_aux;
            max_vol = vol(idx_VCA);
        end   
    end
    
    case   basis_type == 'SVD'
%     Learn the subspace with SVD
%     Ry = Yh(:, mask(1,:) > 0)*Yh(:, mask(1,:) > 0)'/np;
    Ry = Yh;
    [E, ~] = svds(Ry,p);

end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% III. ADMM/SALSA.                                                      %
% ----------------                                                      %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% Auxiliary matrices
IE = E'*E+mu*eye(p);
yyh = E'*Yh_up;

IRE = lambda_m*E'*(R'*R)*E+mu*eye(p);
Ym = im2mat(Ymim);
yym = E'*R'*Ym;

% Define and initialize variables
% All variables (X,V1,V2,V3,D1,D2,D3) will be represented as matrices
X = zeros(nl*nc,p)';
V1 = X;
D1 = X;
V2 = X;
D2 = X;
V3 = X;
D3 = X;
V4 = X;
D4 = X;

% costf = [];

% Initialize tic
t1 = tic; t2 = toc(t1); t1 = tic; t2 = toc(t1);
t1 = tic;

for i=1:iters
    
    %   min   ||XB - V1 - A1||_F^2  +
    %    X    ||X  - V2 - A2||_F^2
    %         ||XDH - V3 - A3||_F^2 +
    %         ||XDV - V4 - A4||_F^2
    %
    X = ConvC(V1+D1, IBD_B, nl) + ConvC(V2+D2, IBD_II, nl) + ...
        ConvC(V3+D3, IBD_DH, nl) +  ConvC(V4+D4, IBD_DV, nl);
    
    
    %  max (1/2)||Yh - EV1M|_F^2 + (mu/2)||XB - V1 - D1||_F^2
    %   V1
    NU1 =  ConvC(X, FB, nl) - D1;
    V1 = IE\(yyh + mu*NU1).*mask + NU1.*(1-mask);
    
    
    %  max (lambda_m/2)||Ym - REV2|_F^2 + (mu/2)||X - V2 - D2||_F^2
    %   V1
    NU2 =  X - D2;
    V2 = IRE\(lambda_m*yym + mu*NU2);
    
    
    % min lambda VTV(V2,V3) + (mu/2)||XDH - V3 - D3||_F^2 + (mu/2)||XDV - V4 - D4||_F^2
    % V2,V3
    NU3 =  ConvC(X, FDH, nl) - D3;
    NU4 =  ConvC(X, FDV, nl) - D4;
    [V3,V4] = vector_soft_col_iso(NU3,NU4,lambda_phi/mu);
    
%     fprintf('iter = %d out of %d\n', i, iters);
        
    % Update Lagrange multipliers
    D1 = -NU1 + V1;    % D1 - (XB - V1)
    D2 = -NU2 + V2;    % D2 - (XDH - V2)
    D3 = -NU3 + V3;    % D3 - (XDV - V3)
    D4 = -NU4 + V4;    % D3 - (XDV - V3)

end
t2 = toc(t1);
fprintf('The algorithm took %2.2f seconds to run.\n', t2);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% IV. Postprocessing.                                                   %
% -------------------------------                                       %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

Zhat = E*X;
Zimhat = mat2im(Zhat, nl);
