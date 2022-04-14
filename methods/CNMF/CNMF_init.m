function Out = CNMF_init(xdata,ydata,w,M,hyper,multi,delta,I_in,delta_h,delta_m,srf,init_mode,mask)
%--------------------------------------------------------------------------
% COUPLED NONNEGATIVE MATRIX FACTORIZATION (CNMF)
% 
% Copyright (c) 2015 Naoto Yokoya All rights reserved.
% Email: yokoya@sal.rcast.u-tokyo.ac.jp
% Update: 2015/01/14
%
% Reference:
% [1] N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix 
%     factorization unmixing for hyperspectral and multispectral data fusion," 
%     IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.
% [2] N. Yokoya, T. Yairi, and A. Iwasaki, "Hyperspectral, multispectral, 
%     and panchromatic data fusion based on non-negative matrix factorization," 
%     Proc. WHISPERS, Lisbon, Portugal, Jun. 6-9, 2011.
%
% This function is an initilization function of CNMF.
%
% USAGE
%       Out = CNMF_init(xdata,ydata,w,M,hyper,multi,delta,ite_max,delta_h,delta_m,srf,mask)
%
% INPUT
%       xdata           : image height
%       ydata           : image width
%       w               : multiple difference of ground sampling distance (scalar)
%       M               : Number of endmembers
%       hyper           : Low-spatial-resolution HS image (band, xdata/w*ydata/w)
%       multi           : MS image (multi_band, xdata*ydata)
%       delta           : Parameter of sum to one constraint
%       I_in            : Maximum number of inner iteration
%       delta_h         : Parameter for HS unmixing
%       delta_m         : Parameter for MS unmixing
%       srf             : Relative specctral response function
%       init_mode       : Initialization mode (0: const, 1: fcls)
%       mask            : (optional) Binary mask for processing (xdata/w,ydata/w)
%
% OUTPUT
%       Out.hyper       : Low-spatial-resolution HS image with ones (band+1, xdata/w*ydata/w)
%       Out.multi       : MS image with ones (multi_band+1, xdata*ydata)
%       Out.W_hyper     : HS endmember matrix with ones (band+1, M)
%       Out.H_hyper     : HS abundance matrix (M, xdata/w*ydata/w)
%       Out.W_multi     : MS endmember matrix with ones (multi_band+1, M)
%       Out.H_multi     : MS abundance matrix (M, xdata*ydata)
%       Out.RMSE_h      : RMSE of HS unmixing
%       Out.RMSE_m      : RMSE of MS unmixing
%
%--------------------------------------------------------------------------

MIN_MS_BANDS = 3;
verbose = 'off'; % default

% masking mode
if nargin == 12
    masking = 0;
elseif nargin == 13
    masking = 1;
else
    disp('Please check the usage of CNMF_fusioin.m');
end

% Initialize W_hyper: band*M
band = size(hyper,1);
multi_band = size(multi,1);
hx = xdata/w;
hy = ydata/w;
if strcmp (verbose, 'on'), disp('Initialize Wh by VCA'); end
[W_hyper, ~] = vca(hyper, M);

% Initialize H_hyper: (M, N_h)
switch masking
    case 0
        H_hyper = ones(M,hx*hy)/M;
    case 1
        H_hyper = ones(hx*hy,M)/M; 
        H_hyper = H_hyper(mask,:)';
end

if init_mode == 1
    H_hyper = fcls( hyper, W_hyper );
end
        
W_hyper = [W_hyper; delta*ones(1,size(W_hyper,2))];
hyper = [hyper; delta*ones(1,size(hyper,2))];

% NMF for Vh 1st
if strcmp (verbose, 'on'), disp('NMF for Vh (1)'); end
for i = 1:I_in
    if i==1
        cost0 = 0;
        for q = 1:I_in*3
            % Update H_hyper
            H_hyper_old = H_hyper;
            H_hyper_n = (W_hyper')*hyper;
            H_hyper_d = (W_hyper')*W_hyper*H_hyper;
            H_hyper = H_hyper.*H_hyper_n./H_hyper_d;
            cost = sum(sum((hyper(1:band,:)-W_hyper(1:band,:)*H_hyper).^2));
            if q>1 && (cost0-cost)/cost<delta_h
                if strcmp (verbose, 'on'),
                    disp(['Initialization of H_hyper converged at the ' num2str(q) 'th iteration '  num2str((cost0-cost)/cost)]);
                end
                H_hyper = H_hyper_old;
                break;
            end
            cost0 = cost;
        end
    else
        % Update W_hyper
        W_hyper_old = W_hyper;
        W_hyper_n = hyper(1:band,:)*(H_hyper');
        W_hyper_d = W_hyper(1:band,:)*H_hyper*(H_hyper');
        W_hyper(1:band,:) = W_hyper(1:band,:).*W_hyper_n./W_hyper_d;
        % Update H_hyper
        H_hyper_old = H_hyper;
        H_hyper_n = (W_hyper')*hyper;
        H_hyper_d = (W_hyper')*W_hyper*H_hyper;
        H_hyper = H_hyper.*H_hyper_n./H_hyper_d;
        cost = sum(sum((hyper(1:band,:)-W_hyper(1:band,:)*H_hyper).^2));
        if (cost0-cost)/cost<delta_h
            if strcmp (verbose, 'on'),
                disp(['Optimization of HS unmixing converged at the ' num2str(i) 'th iteration '  num2str((cost0-cost)/cost)]);
            end
            W_hyper = W_hyper_old;
            H_hyper = H_hyper_old;
            break;
        end
        cost0 = cost;
    end
end

Out.RMSE_h = (sum(sum((hyper(1:band,:)-W_hyper(1:band,:)*H_hyper).^2))/(size(hyper,2)*band))^0.5;
if strcmp (verbose, 'on'), disp(['    RMSE(Vh) = ' num2str(Out.RMSE_h)]); end

% initialize W_multi: (multi_band, M)
W_multi = srf*W_hyper(1:band,:);
W_multi = [W_multi; delta*ones(1,M)];
multi = [multi; delta*ones(1,size(multi,2))];

% initialize H_multi by interpolation
switch masking
    case 0
        H_multi = ones(M,xdata*ydata)/M;
        for i = 1:M
            tmp = imresize(reshape(H_hyper(i,:),hx,hy),w);
            H_multi(i,:) = reshape(tmp,1,[]);
        end
        H_multi(H_multi<0) = 0;
        cost0 = 0;
    case 1
        mask2 = imresize(mask,w,'nearest');
        H_multi = ones(M,size(multi,2))/M;
        for i = 1:M
            tmp = zeros(hx,hy);
            tmp(mask) = H_hyper(i,:);
            tmp = imresize(tmp,w); 
            H_multi(i,:) = reshape(tmp(mask2),1,[]);
        end
        H_multi(H_multi<0) = 0;
        cost0 = 0;
end

% NMF for Vm 1st
if strcmp (verbose, 'on'), disp('NMF for Vm (1)'); end
for i = 1:I_in
    if i==1
        cost0 = 0;
        for q = 1:I_in
            % Update H_multi
            H_multi_old = H_multi;
            H_multi_n = (W_multi')*multi;
            H_multi_d = (W_multi')*W_multi*H_multi;
            H_multi = H_multi.*H_multi_n./H_multi_d;
            cost = sum(sum((multi(1:multi_band,:)-W_multi(1:multi_band,:)*H_multi).^2));
            if i>1 && (cost0-cost)/cost<delta_m
                if strcmp (verbose, 'on'),
                    disp(['Initialization of H_multi converged at the ' num2str(q) 'th iteration ' num2str((cost0-cost)/cost)]);
                end
                H_multi = H_multi_old;
                break;
            end
            cost0 = cost;
        end
    else
        % Update W_multi
        W_multi_old = W_multi;
        if multi_band > MIN_MS_BANDS
            W_multi_n = multi(1:multi_band,:)*(H_multi');
            W_multi_d = W_multi(1:multi_band,:)*H_multi*(H_multi');
            W_multi(1:multi_band,:) = W_multi(1:multi_band,:).*W_multi_n./W_multi_d;
        end
        % Update H_multi
        H_multi_old = H_multi;
        H_multi_n = (W_multi')*multi;
        H_multi_d = (W_multi')*W_multi*H_multi;
        H_multi = H_multi.*H_multi_n./H_multi_d;
        cost = sum(sum((multi(1:multi_band,:)-W_multi(1:multi_band,:)*H_multi).^2));
        if (cost0-cost)/cost<delta_m
            if strcmp (verbose, 'on'),
                disp(['Optimization of MS unmixing converged at the ' num2str(i) 'th iteration '  num2str((cost0-cost)/cost)]);
            end
            W_multi = W_multi_old;
            H_multi = H_multi_old;
            break;
        end
        cost0 = cost;
    end    
end

Out.RMSE_m = (sum(sum((multi(1:multi_band,:)-W_multi(1:multi_band,:)*H_multi).^2))/(size(multi,2)*multi_band))^0.5;
if strcmp (verbose, 'on'), disp(['    RMSE(Vm) = ' num2str(Out.RMSE_m)]); end

Out.hyper = hyper;
Out.multi = multi;
Out.W_hyper = W_hyper;
Out.H_hyper = H_hyper;
Out.W_multi = W_multi;
Out.H_multi = H_multi;
