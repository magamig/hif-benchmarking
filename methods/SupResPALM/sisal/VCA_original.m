function [Ae, indice, Rp] = vca(R,varargin)

% Vertex Component Analysis Algorithm [VCA]
%
% [VCA] J. Nascimento and J. Bioucas-Dias
% "Vertex component analysis: a fast algorithm to unmix hyperspectral data"
% IEEE Transactions on Geoscience and Remote Sensing,  vol. 43, no. 4, 
% pp. 898-910, 2005.
%
% -------------------------------------------------------------------
% Usage:
%
% [Ae, indice, Rp ]= vca(R,'Endmembers',p,'SNR',r,'verbose',v)
%
% ------- Input variables -------------------------------------------
%
%  R - matrix with dimensions L(channels) x N(pixels)
%      Each pixel is a linear mixture of p endmembers
%      signatures R = M X, where M  and X are the mixing matrix 
%      and the abundance fractions matrix, respectively.
%
% 'Endmembers'
%          p - number of endmembers in the scene
%
% ------- Output variables -------------------------------------------
%
% A      - estimated mixing matrix (endmembers signatures)
%
% indice - pixels chosen to be the most pure
%
% Rp     - Data R projected on the identified signal subspace
%
% ------- Optional parameters -----------------------------------------
%
% 'SNR'     - (double) signal to noise ratio (dB)
%             SNR is used to decide the type of projection: projective
%             or orthogonal.
%
% 'verbose' - [{'on'} | 'off']
% ---------------------------------------------------------------------
%
% Please see [VCA] for more details or contact the Authors
%
% -----------------------------------------------------------------------
% version: 3.0 (21-January-2012)
%
% Modifications w.r.t. version 2.1:
%     
%  - Increased efficiency in the memory usage
%  - Correction of a bug in SNR estimation
%  - detection of outliers in the projective projection
%
% -----------------------------------------------------------------------
% Copyright (2012):  José Nascimento (zen@isel.pt)
%                    José Bioucas Dias (bioucas@lx.it.pt)
%
% affineProj is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

verbose = 'on'; % default
snr_input = 0;  % estimate the SNR
p = 0;          % default number of endmembers


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim_in_par = length(varargin);
if (nargin - dim_in_par)~=1
    error('Wrong parameters');
elseif rem(dim_in_par,2) == 1
    error('Optional parameters should always go by pairs');
else
    for i = 1 : 2 : (dim_in_par-1)
        switch lower(varargin{i})
            case 'verbose'
                verbose = varargin{i+1};
            case 'endmembers'
                p = varargin{i+1};
            case 'snr'
                SNR = varargin{i+1};
                snr_input = 1;       % user input SNR
            otherwise
                fprintf(1,'Unrecognized parameter:%s\n', varargin{i});
        end %switch
    end %for
end %if


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(R)
    error('there is no data');
else
    [L N]=size(R);  % L number of bands (channels)
    % N number of pixels (LxC)
end

if (p<=0 | p>L | rem(p,1)~=0),
    error('ENDMEMBER parameter must be an  integer between 1 and L');
end

if (L-p < p) & (snr_input == 0)
    if strcmp (verbose, 'on'),
        fprintf(1,' i can not  estimate SNR [(no bands)-p < p]\n');
        fprintf(1,' i will apply the projective projection\n');
        snr_input = 1;
        SNR = 100;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stuff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


r_m = mean(R,2);
Corr = R*R'/N;
[Ud,Sd] = svds(Corr,p);            % computes the a p-orth basis


if snr_input == 0
    % estimate SNR
    Pt = trace(Corr);      % total power     (signal + noise)
    Pp = sum(diag(Sd));   % projected power (signal + projected noise)
    Pn = (Pt-Pp)/(1-p/L);       % rough noise power estimate
    if Pn > 0
        SNR = 10*log10(Pp/Pn);  % signal-to-noise ratio in dB
        if strcmp (verbose, 'on'), 
            fprintf(1,'SNR estimated = %g[dB]\n',SNR); 
        end
    else
        SNR = 0;
        if strcmp (verbose, 'on'), 
            fprintf(1,'Input data belongs to a p-subspace'); 
        end
    end
end

% SNR threshold to decide the projection:
%       Projective Projection
%       Projection on the  p-1 subspace
SNR_th = 15 + 10*log10(p);

if SNR < SNR_th,
    if strcmp (verbose, 'on')
        fprintf(1,'Select proj. on the to (p-1) subspace.\n')
        fprintf(1,'I will apply the projective projection\n')
    end
    % orthogonal projection on an the best (p-1) affine set 
    d = p-1;
    Cov  = Corr - r_m*r_m';
    [Ud,Sd] = svds(Cov,d);         % computes the a d-orth basis
    R_o = R - repmat(r_m,[1 N]);   % remove mean 
    x_p =  Ud' * R_o;  % project the zero-mean data onto  a p-subspace
    
    %  original data projected on the indentified subspace 
    Rp =  Ud * x_p(1:d,:) + repmat(r_m,[1 N]);   % again in dimension L
    
    % compute the angles (cosines) between the projected vectors and the
    % original
    cos_angles = sum(Rp.*R)./(sqrt(sum(Rp.^2).*sum(R.^2)));
    
    
    % lift to p-dim
    c = max(sum(x_p.^2,1))^0.5;
    y = [x_p ; c*ones(1,N)] ;
    
else
    if strcmp (verbose, 'on'), 
        fprintf(1,'... Select the projective proj. (dpft)\n');
    end
    
    % projective projection
    d = p;    
    % project into a p-dim subspace (filter noise)
    x_p = Ud'*R;
    
    %  original data projected on the indentified subspace 
    Rp =  Ud * x_p(1:d,:);      % again in dimension L 
    
    
    
    % find a direction orthogonal to the affine set
    u = mean(x_p,2)*p;             
    
    % ensure that angle(xp(:,i),u) is positive

    scale = sum( x_p .* repmat(u,[1 N]) );

    th = 0.01;   % close to zero
    mask = scale < th ; 
    scale = scale.*(1-mask) + mask;
    
    y =  x_p./ repmat(scale,[d 1]) ;
    pt_errors = find(mask);
    
    % replace the bad vectors with a vector in the middle of the simplex
    y(:,pt_errors) = (u/norm(u)^2)*ones(1,length(pt_errors));
    
     
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VCA algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pointers for endmembers
indice = zeros(1,p);
% save endmembers
A = zeros(p,p);
A(p,1) = 1;

for i=1:p
    w = rand(p,1);
    f = w - A*pinv(A)*w;
    f = f / sqrt(sum(f.^2));
    
    v = f'*y;
    [v_max indice(i)] = max(abs(v));
    A(:,i) = y(:,indice(i));        % same as x(:,indice(i))
end
Ae = Rp(:,indice);

return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of the vca function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



