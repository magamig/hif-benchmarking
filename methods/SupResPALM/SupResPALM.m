% Code (c) Charis Lanaras, ETH Zurich, Oct 28 2015 (see LICENSE)
% charis.lanaras@geod.baug.ethz.ch

function [E,A] = SupResPALM(hyper, multi, srf, p, h)
% SupResPALM - Perform hyperspectral super-resolution by spectral unmixing

% Usage
%   [E,A] = SupResPALM(hyper, multi, truth, srf, p, h )
% 
% Inputs
%   hyper - hyperspectral image (in 2D format)
%   multi - RGB or multispectral image (in 2D format)
%   truth - ground truth image (in 2D format) - used only for evaluation
%   srf   - the spectral response function of the RGB multispectral camera
%   p     - the desired number of material spectra to extract
%   h     - (optional) the hight of the image, in case of non-square images
%           (h must be in accordance with the spatial downsampling factor)
%
% Outputs
%   E - Matrix of endmembers (spectral basis)
%   Y - Abundances (mixing coefficients)
%
% References
%   C. Lanaras, E. Baltsavias, K. Schindler. Hyperspectral Super-Resolution
%   by Coupled Spectral Unmixing. In: ICCV 2015
% 
% Comment
%    To transform a 3D image cube to the respective 2D format, you can use
%    the hyperConvert2d.m function.
%

if ndims(hyper)==3
    hyper = hyperConvert2d(hyper);
end
if ndims(multi)==3
    multi = hyperConvert2d(multi);
end
%if ndims(truth)==3
%    truth = hyperConvert2d(truth);
%end
%if ~(nargin==6)
%    h = sqrt(size(multi,2));
%end

maxIter = 200;
epsilon = 0.0001;
s2=p/2; %Active endmemebers per pixel (on average over the whole image)
% Default value
% To activate this option you need to uncomment the specified line in
%   highResStep.m
resAB(1) = 1;

scale_diff = sqrt(size(multi,2)/size(hyper,2)); % difference of resolution
[S St] = hyperSpatialDown(h, size(multi,2)/h, scale_diff);

% Initialisations
E = sisal(hyper,p, 'spherize', 'no','MM_ITERS',80, 'TAU', 0.006, 'verbose',0);
%[ E, ~ ] = vca( hyper, p );

AS = sunsal(E, hyper,'POSITIVITY','yes','ADDONE','yes');
A = hyperConvert2d(imresize(hyperConvert3d(AS,h/scale_diff),scale_diff));


for j=1:maxIter

    % hyperspectral least-squares
    [E, ~, res] = lowResStep(hyper,E,AS);
    resA2(j) = min(res);
    
    % spectraly downgrade endmemebrs
    RE = srf*E;
    
    % multispectral least-squares
    [~, A, res] = highResStep(multi,RE,A,s2);
    resB2(j) = min(res);
    
    % update abundances for hyperspectral step
    %AS = A*S;
    AS = reshape(gaussian_down_sample(reshape(A',h,size(A,2)/h,[]),scale_diff),size(hyper,2),[])'; % modified for HSMSFusionToolbox
    
    % Residual of the objective function (5a)
    resAB(j+1) = resA2(j)+resB2(j);
    
    % Compute RMSE only for printing during procedure
    %RMSE(j) = hyperErrRMSE(truth,E*A);
    
    % Convergence checks
    %if ( resAB(j) / resAB(j+1) ) > 1+epsilon || ( resAB(j) / resAB(j+1) ) < 1-epsilon
    %    fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
    %else
    %    fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
    %    fprintf(['Stopped after ' num2str(j) ' iterations. Final RMSE: ' num2str(RMSE(j)) '\n'])
    %    break
    %end    
    
end
end

function [ E, A, res ] = highResStep( M, E, A, sparse_factor )
% Solving eq. (7) with a projected gradient descent method.

maxIter = 100;
epsilon = 1.01;
gamma2 = 1.01;

N = size(M, 2);
beta = round(sparse_factor*N); % Number of desirable non-zero entries

res(1) = norm(M-E*A,'fro')+100;

for k=1:maxIter
    E_old = E;
    A_old = A;
    
    % 2.2. Update the Abundances
    dk = gamma2 * norm( E*E' ,'fro');
    V = A - 1/dk * E' * ( E*A - M );
    
    % Uncomment Tau_multi and comment the following line to use the sparse
    % constraint
    % A = Tau_multi(Pplusb(V),beta);
    A = Pplusb(V);
    
    % Calculation of residuals
    res(k+1,1) = sqrt(norm(M-E*A,'fro')^2/size(M,1)/size(M,2));
    
    % Checks for exiting iteration
    if (1/res(k+1) * res(k)) < epsilon
        fprintf(['Multi: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        E = E_old;
        A = A_old;
        text = ['Multi: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
        disp(text)
        break
    end
    
end
end

function [ E, A, res ] = lowResStep( H, E, A )
% Solving eq. (6) with a projected gradient descent method.

maxIter = 100;
epsilon = 1.01;
gamma1 = 1.01;

notfirst = 0;

res(1) = norm(H-E*A,'fro')+100;

for k=1:maxIter
    E_old = E;
    A_old = A;
    
    % 2.1. Update of signatures
    ck = gamma1 * norm(A*A','fro');
    U = E - 1/ck * ( E*A - H ) * A';
    E = Pplusa(U);
    
    % Calculation of residuals
    res(k+1,1) = sqrt(norm(H-E*A,'fro')^2/size(H,1)/size(H,2));
    
    % Checks for exiting iteration
    if (1/res(k+1) * res(k)) < epsilon
        fprintf(['Hyper: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        if notfirst == 1
            E = E_old;
            A = A_old;
            text = ['Hyper: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
            disp(text)
            break
        else
            notfirst = 1;
        end
    end
    
end
end

function U = Pplusa(U)
% max{0,U}
U(U<0) = 0;
U(U>1) = 1;
end

function V = Pplusb(V)
% Simplex Projection

V = hyperConvert3d(V,2);
V1 = reproject_simplex_mex_fast(V);
V = hyperConvert2d(V1);

end

function U = Tau_multi(U,s)

% keep only the first s largest entries of U
U1 = reshape(U,[],1);
[values, ind] = sort(U1,'descend');
U1 = zeros(length(U1),1);
U1(ind(1:s),1) = values(1:s);
U = reshape(U1,size(U));

end