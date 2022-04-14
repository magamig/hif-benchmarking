function Out = CNMF_ite(xdata,ydata,w,M,hyper,multi,W_hyper,H_hyper,W_multi,H_multi,I_in,delta_h,delta_m,I_out,i_out,srf,mask)
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
% This function is an iteration function of CNMF.
%
% USAGE
%       Out = CNMF_ite(xdata,ydata,w,M,hyper,multi,W_hyper,H_hyper,W_multi,H_multi,ite_max,delta_h,delta_m,iter,srf,mask)
%
% INPUT
%       xdata           : image height
%       ydata           : image width
%       w               : multiple difference of ground sampling distance (scalar)
%       M               : Number of endmembers
%       hyper           : Low-spatial-resolution HS image (band, xdata/w*ydata/w)
%       multi           : MS image (multi_band, xdata*ydata)
%       W_hyper         : HS endmember matrix with ones (band+1, M)
%       H_hyper         : HS abundance matrix (M, xdata/w*ydata/w)
%       W_multi         : MS endmember matrix with ones (multi_band+1, M)
%       H_multi         : MS abundance matrix (M, xdata*ydata)
%       delta           : Parameter of sum to one constraint
%       I_in            : Maximum number of inner iteration
%       delta_h         : Parameter for HS unmixing
%       delta_m         : Parameter for MS unmixing
%       I_out           : Maximum number of outer iteration
%       i_out           : Current number of outer iteration
%       srf             : Relative specctral response function
%       mask            : (optional) Binary mask for processing (xdata/w,ydata/w)
%
% OUTPUT
%       Out.hyper       : Low-spatial-resolution HS image with ones (band+1, xdata/w*ydata/w)
%       Out.multi       : MS image with ones (multi_band+1, xdata*ydata)
%       Out.W_hyper     : HS endmember matrix with ones (band+1, M)
%       Out.H_hyper     : HS abundance matrix (M, xdata/w*ydata/w)
%       Out.W_multi     : MS endmember matrix with ones before MS unmixing (multi_band+1, M)
%       Out.H_multi     : MS abundance matrix before MS unmixing (M, xdata*ydata)
%       Out.W_multi2    : MS endmember matrix with ones after MS unmixing (multi_band+1, M)
%       Out.H_multi2    : MS abundance matrix after MS unmixing (M, xdata*ydata)
%       Out.RMSE_h      : RMSE of HS unmixing
%       Out.RMSE_m      : RMSE of MS unmixing
%
%------------------------------------------------------------------

MIN_MS_BANDS = 3;
verbose = 'off'; % default

% masking mode
if nargin == 16
    masking = 0;
elseif nargin == 17
    masking = 1;
else
    disp('Please check the usage of CNMF_fusioin.m');
end

if strcmp (verbose, 'on'), disp(['Iteration' num2str(i_out)]); end
hx = xdata/w;
hy = ydata/w;
band = size(hyper,1)-1; % size(hyper,1)-1
multi_band = size(multi,1)-1; % size(multi,1)-1
% Initialize H_hyper form H_multi
switch masking
    case 0
        H_hyper = reshape(gaussian_down_sample(reshape(H_multi',xdata,ydata,M),w),[],M)';
    case 1
        mask2 = imresize(mask,w,'nearest');
        for q = 1:M
            tmp = zeros(xdata,ydata);
            tmp(mask2) = H_multi(q,:);
            tmp = reshape(gaussian_down_sample(tmp,w),1,[]);
            H_hyper(q,:) = reshape(tmp(mask),1,[]);
        end
end

% NMF for Vh
if strcmp (verbose, 'on'), disp(['NMF for Vh (' num2str(i_out+1) ')']); end
for i = 1:I_in
    if i==1
        cost0 = 0;
        for q = 1:I_in
            % Update W_hyper
            W_hyper_old = W_hyper;
            W_hyper_n = hyper(1:band,:)*(H_hyper');
            W_hyper_d = W_hyper(1:band,:)*H_hyper*(H_hyper');
            W_hyper(1:band,:) = W_hyper(1:band,:).*W_hyper_n./W_hyper_d;
            cost = sum(sum((hyper(1:band,:)-W_hyper(1:band,:)*H_hyper).^2));
            if q>1 && (cost0-cost)/cost<delta_h
                if strcmp (verbose, 'on')
                    disp(['Initialization of W_hyper converged at the ' num2str(q) 'th iteration '  num2str((cost0-cost)/cost)]);
                end
                W_hyper = W_hyper_old;
                break;
            end
            cost0 = cost;
        end
    else
        % Update H_hyper
        H_hyper_old = H_hyper;
        if multi_band > MIN_MS_BANDS % added
            H_hyper_n = (W_hyper')*hyper;
            H_hyper_d = (W_hyper')*W_hyper*H_hyper;
            H_hyper = H_hyper.*H_hyper_n./H_hyper_d;
        end
        % Update W_hyper
        W_hyper_old = W_hyper;
        W_hyper_n = hyper(1:band,:)*(H_hyper');
        W_hyper_d = W_hyper(1:band,:)*H_hyper*(H_hyper');
        W_hyper(1:band,:) = W_hyper(1:band,:).*W_hyper_n./W_hyper_d;      
        cost = sum(sum((hyper(1:band,:)-W_hyper(1:band,:)*H_hyper).^2));
        if (cost0-cost)/cost<delta_h
            if strcmp (verbose, 'on')
                disp(['Optimization of HS unmixing converged at the ' num2str(i) 'th iteration ' num2str((cost0-cost)/cost)]);
            end
            H_hyper = H_hyper_old;
            W_hyper = W_hyper_old;
            break;
        end
        cost0 = cost;
    end
end

Out.RMSE_h = (sum(sum((hyper(1:band,:)-W_hyper(1:band,:)*H_hyper).^2))/(size(hyper,2)*band))^0.5;
if strcmp (verbose, 'on'), disp(['    RMSE(Vh) = ' num2str(Out.RMSE_h)]); end

Out.W_hyper = W_hyper;
Out.H_hyper = H_hyper;
Out.W_multi = W_multi;
Out.H_multi = H_multi;

if I_out > 1
    %% Initialize W_multi and H_multi
    W_multi(1:multi_band,:) = srf*W_hyper(1:band,:);

    % NMF for Vm
    if strcmp (verbose, 'on'), disp(['NMF for Vm (' num2str(i_out+1) ')']); end
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
                if q>1 && (cost0-cost)/cost<delta_m
                    if strcmp (verbose, 'on')
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
                if strcmp (verbose, 'on')
                    disp(['Optimization of MS unmixing converged at the ' num2str(i) 'th iteration ' num2str((cost0-cost)/cost)]);
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

    Out.W_multi2 = W_multi;
    Out.H_multi2 = H_multi;
end
