function out = vd(data,alpha)
%--------------------------------------------------------------------------
% Virtual dimensionality
%
% USAGE
%   out = vd(data,alpha)
% INPUT
%   data : HSI data (bands,pizels)
%   alpha: False alarm rate
% OUTPUT
%   out  : Number of spectrally distinct signal sources in data
%
% REFERENCE
% [1] J. Harsanyi, W. Farrand, and C.-I Chang, "Determining the number and 
%     identity of spectral endmembers: An integrated approach using 
%     Neyman-Pearson eigenthresholding and iterative constrained RMS error 
%     minimization," in Proc. 9th Thematic Conf. Geologic Remote Sensing, 
%     Feb. 1993.
% [2] Chang, C.-I. and Du, Q., "Estimation of number of spectrally distinct 
%     signal sources in hyperspectral imagery," IEEE Transactions on Geoscience
%     and Remote Sensing, vol. 42, pp. 608-619, 2004.
%--------------------------------------------------------------------------

if (nargin < 2)
    alpha = 10^(-3);
end

[L N] = size(data);

R = (data*data')/N;
K = cov(data');

e_r = sort(eig(R),'descend');
e_k = sort(eig(K),'descend');

diff = e_r - e_k;
variance = (2*(e_r.^2+e_k.^2)/N).^0.5;

tau = -norminv(alpha,zeros(L,1),variance);
out = sum(diff > tau);
