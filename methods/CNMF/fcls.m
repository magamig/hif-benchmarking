function [ A ] = fcls( X, S )
%--------------------------------------------------------------------------
% Fully constrained least squares
%
% USAGE
%   [ A ] = hyperFcls( X, S )
% INPUT
%   X  : HSI data (bands,pixels)
%   S  : Matrix of endmembers (bands,p)
% OUTPUT
%   A  : Matrix of abundance maps (num of endmembers,pixels)
%
% REFERENCE
% [1] D. C. Heinz and C.-I. Chang, "Fully constrained least squares linear
%     spectral mixture analysis method for material quantification in 
%     hyperspectral imagery," IEEE Trans. Geosci. Remote Sens., vol. 39, no. 3,
%     pp. 529 - 545, Mar. 2001.
% [2] N. Keshava and J. F. Mustard, "Spectral unmixing," IEEE Signal Processing
%     Magazine, pp. 44 - 57, Jan. 2002.
%--------------------------------------------------------------------------

if (~ismatrix(S))
    error('X must be a bands x pixels matrix.');
end

[bands, pixels] = size(X);
[bands2, numofems] = size(S);
if (bands ~= bands2)
    error('X and S must have the same number of spectral bands.');
end

A = zeros(numofems, pixels);
S_ori = S;
for n = 1:pixels
    c = numofems;
    done = 0;
    ref = 1:numofems;
    x = X(:, n);
    S = S_ori;
    while not(done)
        a_ls_hat = (S'*S)\S'*x;
        s = ((S')*S)\ones(c, 1);
        a_fcls_hat = a_ls_hat - (S'*S)\ones(c, 1)*((ones(1, c)*((S'*S)\ones(c, 1)))\(ones(1, c)*a_ls_hat-1));
        if (sum(a_fcls_hat>0) == c)
            alpha = zeros(numofems, 1);
            alpha(ref) = a_fcls_hat;
            break;
        end
        idx = find(a_fcls_hat<0);
        a_fcls_hat(idx) = a_fcls_hat(idx) ./ s(idx);
        [~, maxIdx] = max(abs(a_fcls_hat(idx)));
        maxIdx = idx(maxIdx);
        alpha(maxIdx) = 0;
        kIdx = setdiff(1:size(S, 2), maxIdx);
        S = S(:, kIdx);
        c = c - 1;
        ref = ref(kIdx);
    end
    A(:, n) = alpha;
end

