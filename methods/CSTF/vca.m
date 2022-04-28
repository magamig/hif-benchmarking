function [ U, indicies ] = vca( R, p )
%--------------------------------------------------------------------------
% Vertex Component Analysis algorithm
%
% USAGE
%   [ U, indicies ] = vca( R, p )
% INPUT
%   R  : HSI data (bands,pixels)
%   p  : Number of endmembers
% OUTPUT
%   U  : Matrix of endmembers (bands,p)
%   indicies : Indicies of endmembers in R
%
% REFERENCE
% J. M. P. Nascimento and J. M. B. Dias, "Vertex component analysis: A 
% fast algorithm to unmix hyperspectral data," IEEE Transactions on 
% Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898 - 910, Apr. 2005.
%--------------------------------------------------------------------------

[L, N]=size(R);

% Estimate SNR
r_m = mean(R,2);      
R_o = R - repmat(r_m,[1 N]);
[Ud,~,~] = svds(R_o*R_o'/N,p);  % computes the p-projection matrix 
x_p =  Ud' * R_o;                 % project the zero-mean data onto p-subspace
P_y = sum(R(:).^2)/N;
P_x = sum(x_p(:).^2)/N + r_m'*r_m;
SNR = abs(10*log10( (P_x - p/L*P_y)/(P_y- P_x) ));

%fprintf('SNR estimate [dB]: %g\n', SNR);

% Determine which projection to use.
SNRth = 15 + 10*log(p) + 8;
%SNRth = 15 + 10*log(p); % threshold proposed in the original paper
if (SNR > SNRth) 
    d = p;
    [Ud,~,~] = svds(R*R'/N,d);
    X = Ud'*R;
    u = mean(X, 2);
    Y =  X ./ repmat( sum( X .* repmat(u,[1 N]) ) ,[d 1]);
else
    d = p-1;
    r_m = mean(R,2);      
    R_m = repmat(r_m,[1 N]); % mean of each band
    R_o = R - R_m; 
    [Ud,~,~] = svds(R_o*R_o'/N,d);
    X =  Ud'*R_o;
    X = X(1:d,:);
    c = max(sum(X.^2,1))^0.5;
    Y = [X ; c*ones(1,N)] ;
end

e_u = zeros(p, 1);
e_u(p) = 1;
A = zeros(p, p);
A(:, 1) = e_u;
indicies = zeros(1,p);
for i=1:p
    w = rand(p, 1);
    f = w - A*pinv(A)*w;
    f = f / sqrt(sum(f.^2));

    v = f.'*Y; 
    [~, indicies(i)] = max(abs(v));
    A(:,i) = Y(:,indicies(i));
end

if (SNR > SNRth)
    U = Ud*X(:,indicies);
else
    U = Ud*X(:,indicies) + repmat(r_m,[1 p]);
end

