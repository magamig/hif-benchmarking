%% demo_sisal_noise_comparison
%
% This demo illustrates the  robustnes of SISAL [1] to noise in
% comparion with the MVSA [2] (hard version) and VCA [3] algorithms
%
% SISAL: Simplex identification via split augmented Lagrangian
%
% [1] J. Bioucas-Dias, "A variable splitting augmented Lagrangian approach
%     to linear spectral unmixing", in  First IEEE GRSS Workshop on
%     Hyperspectral Image and Signal Processing-WHISPERS'2009, Grenoble,
%     France,  2009. Available at http://arxiv.org/abs/0904.4635v
%
% MVSA: Minimum volume simplex analysis
%   
% [2] Jun Li and José M. Bioucas-Dias
%     "Minimum volume simplex analysis: A fast algorithm to unmix hyperspectral data"
%      in IEEE International Geoscience and Remote sensing Symposium
%      IGARSS’2008, Boston, USA,  2008.
%
%
% NOTE:  VCA (Vertex Component Analysis) [3] is used to initialize SISAL. However,
%        VCA is a pure-pixel based algorithm and thus it is not suited to
%        the data sets herein considered.  Nevertheless, we plot VCA results,
%        to highlight the advantage of non-pure-pixel based algorithms over the
%        the pure-pixel based ones.
%
%
% VCA: Vertex component analysis
%
% [3] J. Nascimento and J. Bioucas-Dias, "Vertex componet analysis",
%     IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4,
%     pp. 898-910, 2005.
%
%
% Authors: Jose M. Bioucas-Dias (December, 2009)
%
%
%
%% -------------- PROBLEM DESCRIPTION ------------------------------------
%
%
%   SISAL estimates the vertices  M={m_1,...m_p} of the
%  (p-1)-dimensional   simplex of minimum volume containing the vectors
%  [y_1,...y_N].
%
%   SISAL assumes that y belongs to an affine space. It may happen,
%   however, that the data supplied by the user is not in an affine
%   set. For this reason, the first step this code implements
%   is the estimation of the affine set the best represent
%   (in the l2 sense) the data.
%
%   Any vector y_i   belongs  thus to the   convex hull of  the
%   columns of M; i.e.,
%
%                   y_i = M*x_i
%
%  where x_i belongs to the probability (p-1)-simplex.
%

%% beginning of the demo

clear all,
clc
close all
verbose= 1;


%%
%--------------------------------------------------------------------------
%        Simulation parameters
%-------------------------------------------------------------------------
% p                         -> number of endmembers
% N                         -> number of pixels
% SNR                       -> signal-to-noise ratio (E ||y||^2/E ||n||^2) in dBs
% SIGNATURES_TYPE           -> see below
% L                         -> number of bands (only valid for SIGNATURES_TYPE = 5,6)
% COND_NUMBER               -> conditioning number of the mixing matrix (only for SIGNATURES_TYPE = 5,6)
% DECAY                     -> singular value decay rate
% SHAPE_PARAMETER           -> determines the distribution of spectral
%                              over the simplex (see below)
% MAX_PURIRY                -> determines the maximum purity of  the
%                              mixtures.  If  MAX_PURIRY < 1, ther will be
%                              no pure pixels is the data


% SIGNATURES_TYPE;
%  1 - sampled from the USGS Library
%  2 - not available
%  3 - random (uniform)
%  4 - random (Gaussian)
%  5 - diagonal with conditioning number COND_NUMBER and DECAY exponent
%  6 - fully populated with conditioning number COND_NUMBER and DECAY exponent

% NOTE: For SIGNATURES_TYPE 5 or 6, the difficulty of the problem is
% determined by the parameters
% COND_NUMBER
% DECAY


% Souces are Dirichlet distributed
% SHAPE_PARAMETER ;
%   = 1   -  uniform over the simplex
%   > 1   -  samples moves towards the center
%            of the simplex, corresponding to highly mixed materials and thus
%            the unmixing is  more difficult
%   ]0,1[ -  samples moves towards the facets
%            of the simplex. Thus the unmixing is easier.


%
%--------------------------------------------------------------------------
%        SELECTED PARAMETERS FOR AN EASY PROBLEM
%-------------------------------------------------------------------------

SIGNATURES_TYPE = 1;        % Uniform in [0,1]
p = 3;                      % number of endmembers
N = 2000;                   % number of pixels
SNR = 5;                   % signal-to-noise ratio (E ||y||^2/E ||n||^2) in dBs
L = 200;                    % number of bands (only valid for SIGNATURES_TYPE = 2,3)
% COND_NUMBER  = 1;           % conditioning number (only for SIGNATURES_TYPE = 5,6)
% DECAY = 1;                  % singular values decay rate  (only for SIGNATURES_TYPE = 5,6)
SHAPE_PARAMETER  = 1   ;    % uniform over the simplex
MAX_PURIRY = 0.8;           % there are pure pixels in the data set
OUTLIERS  = 0;              % Number of outliers in the data set


%%
%--------------------------------------------------------------------------
%        Begin the simulation
%-------------------------------------------------------------------------
switch SIGNATURES_TYPE
    case 1
        rand('seed',5);
        load('USGS_1995_Library')
        wavlen=datalib(:,1);    % Wavelengths in microns
        [L n_materiais]=size(datalib);
        % select randomly
        sel_mat = 4+randperm(n_materiais-4);
        sel_mat = sel_mat(1:p);
        M = datalib(:,sel_mat);
        % print selected endmembers
        %         fprintf('endmembers:\n')
        %         for i=1:p
        %             aux = names(sel_mat(i),:);
        %             fprintf('%c',aux);
        %             st(i,:) = aux;
        %         end
        clear datalib wavelen names aux st;
    case 2
        error('type not available')
    case 3
        M = rand(L,p);
    case 4
        M = randn(L,p);
    case 5
        L=p;
        M = diag(linspace(1,(1/(COND_NUMBER)^(1/DECAY)),p).^DECAY);
    case 6
        L=p;
        M = diag(linspace(1,(1/(COND_NUMBER)^(1/DECAY)),p).^DECAY);
        A = randn(p);
        [U,D,V] = svd(A);
        M = U*M*V';
        clear A U D V;
    otherwise
        error('wrong signatute type')
end


%--------------------------------------------------------------------------
%        Set noise parameters (to be used in spectMixGen function)
%-------------------------------------------------------------------------
% white_noise = [0 1 1];     % white noise
% % non-white noise parameters
% eta   = 10;                % spread of the noise shape
% level = 10;                % floor lever
% gauss_noise = [level L/2 eta]; % Gaussian shaped noise centered at L/2 with spread eta
% % and floor given by level
% rect_noise  = [level L/2 eta]; % Rectangular shaped noise centered at L/2 with spread eta
% % and floor given by level

%%
%--------------------------------------------------------------------------
%        Generate the data set
%-------------------------------------------------------------------------
%
%   Sources are Diriclet distributed (shape is controled by 'Source_pdf' and
%   'pdf_pars': 'Source_pdf' = 'Diri_id' and 'pdf_pars' = 1 means a uniform
%   density over the simplex).  The user may change the parameter to
%   generate other shapes. Mixtures are aldo possible.
%
%   'max_purity' < 1 means that there are no pure pixels
%
randn('seed',5);
[Y,x,noise] = spectMixGen(M,N,'Source_pdf', 'Diri_id','pdf_pars',SHAPE_PARAMETER,...
    'max_purity',MAX_PURIRY*ones(1,p),'no_outliers',OUTLIERS, ...
    'violation_extremes',[1,1.2],'snr', SNR, ...
    'noise_shape','uniform');

%%
%--------------------------------------------------------------------------
%        Remove noise  (optional)
%-------------------------------------------------------------------------
%   noise_hat = estNoise(Y);
%   Y = Y-noise_hat;
%   clear noise_hat




%%
%--------------------------------------------------------------------------
%       Project  on the  affine set defined by the data in the sense L2
%-------------------------------------------------------------------------
%
%   The application of this projection ensures that the data is in
%   an affine set.
%
%   Up is an isometric matrix that spans the subspace where Y lives
[Y,Up,my,sing_val] = dataProj(Y,p,'proj_type','affine');



%%
%--------------------------------------------------------------------------
%        Degree of Difficulty of the problem
%-------------------------------------------------------------------------
%% compute original subspace
sing_vects = svds(M,p);

% Condition number gives an idea of the difficulty in inferring
% the subspace
fprintf('Conditioning number of M = %2f \n', sing_vects(1)/sing_vects(end))
% fprintf('\n Hit any key: \n ');
% pause;

Cx = Up'*(M*x)*(M*x)'*Up/N;
Cn = Up'*noise*noise'*Up/N;
[U,D]=svd(Cx);

% compute the SNR along the direction corresponding the smaller eigenvalue

LOWER_SNR= D(p,p)/(U(:,p)'*Cn*U(:,p));
fprintf('\nSNR along the signal smaller eigenvalue = %f \n', LOWER_SNR);
if LOWER_SNR < 20
    fprintf('\nWARNING: This problem is too hard and the results may be inaccurate \n')
end

clear noise x;



%%
%--------------------------------------------------------------------------
%         ALGORITHMS
%-------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
%         SISAL[1] -  Simplex identification via split augmented Lagrangian
%-------------------------------------------------------------------------

% start timer
tic
% set the hinge regularization parameter
tau = 0.1;
[A_est] = sisal(Y,p, 'spherize', 'yes','MM_ITERS',40, 'TAU',tau, 'verbose',2);
drawnow;
t(1)=toc;
Msisal =  Up'*A_est;




%%
%--------------------------------------------------------------------------
%         VCA [5] - Vertex component analysis
%-------------------------------------------------------------------------
%
% start timer
tic
[Ae, indice,ys]= VCA(Y,'Endmembers',p);
Mvca = Up'*Ae;
% stop timer
t(2) = toc;


%%
%--------------------------------------------------------------------------
%         MVSA [4] - Minimum volume simple analysis
%-------------------------------------------------------------------------

%start timer
tic

A_est = mvsa(Y,p,'spherize','yes');
Mvsa = Up'*A_est;

%stop timer
t(3) = toc;




%%
%--------------------------------------------------------------------------
%         Project the original mixing matxix and the data set the
%         identified affine set.
%-------------------------------------------------------------------------
Mtrue = Up'*M;
Y=Up'*Y;


%%
%--------------------------------------------------------------------------
%        Display the results
%-------------------------------------------------------------------------

% selects axes  to display
%

I = 1;
J = 2;
K = 3;

% canonical orthogonal directions
E_I = eye(p);

v1 = E_I(:,I);
v2 = E_I(:,J);
v3 = E_I(:,K);

% original axes

Q = inv(Mtrue);
% v1 = Q(I,:)';
% v2 = Q(J,:)';
% v3 = Q(K,:)';

Y = [v1 v2 v3]'*Y;
m_true = [v1 v2 v3]'*Mtrue;
m_sisal = [v1 v2 v3]'*Msisal;



% legend
leg_cell = cell(1);
leg_cell{end} = 'data points';
H_2=figure;
plot(Y(1,OUTLIERS+1:end),Y(2,OUTLIERS+1:end),'b.')

hold on;
if OUTLIERS> 0
    plot(Y(1,1:OUTLIERS),Y(2,1:OUTLIERS),'x','Color',[0 0.6 0.6],'LineWidth',3 )
    leg_cell{end +1} = 'OUTLIERS';
end



plot(m_true(1,[1:p 1]), m_true(2,[1:p 1]),'k*')
leg_cell{end +1} = 'true';



plot(m_sisal(1,1:p), m_sisal(2,1:p),'S', 'Color',[1 0 0])
leg_cell{end +1} = 'SISAL';




m_vca  = [v1 v2 v3]'*Mvca;
plot(m_vca(1,[1:p 1]), m_vca(2,[1:p 1]),'p', 'Color',[0  0 0])
leg_cell{end +1} = 'VCA';



m_vsa  = [v1 v2 v3]'*Mvsa;
plot(m_vsa(1,[1:p 1]), m_vsa(2,[1:p 1]),'O','Color',[0 0 0])
leg_cell{end +1} = 'MVSA';



xlabel('v1''*Y'),ylabel('v2''*Y');
legend(leg_cell)
title('Endmembers and data points (2D projection)')

plot(m_true(1,[1:p 1]), m_true(2,[1:p 1]),'k','LineWidth',2)
plot(m_sisal(1,[1:p 1]), m_sisal(2,[1:p 1]),'r','LineWidth',2)
plot(m_vsa(1,[1:p 1]), m_vsa(2,[1:p 1]),'LineWidth',2,'Color',[0.8 0 0.8])

fprintf('\nTIMES (sec):\n SISAL = %3.2f\n MVSA = %3.2f\n VCA = %3.2f\n', t(1),t(3),t(2))

%--------------------------------------------------------------------------
%        Display errors
%-------------------------------------------------------------------------

%% alignament
% soft

angles = Mtrue'*Msisal./(repmat(sqrt(sum(Mtrue.^2)),p,1)'.*(repmat(sqrt(sum(Msisal.^2)),p,1)));
P = zeros(p);
for i=1:p
    [dummy,j] = max(angles(i,:));
    P(j,i) = 1;
    angles(:,j) = -inf;
end
% permute colums
Msisal = Msisal*P;
SISAL_ERR =norm(Mtrue-Msisal,'fro')/norm(Mtrue,'fro');


angles = Mtrue'*Mvca./(repmat(sqrt(sum(Mtrue.^2)),p,1)'.*(repmat(sqrt(sum(Mvca.^2)),p,1)));
P = zeros(p);
for i=1:p
    [dummy,j] = max(angles(i,:));
    P(j,i) = 1;
    angles(:,j) = -inf;
end
% permute colums
Mvca = Mvca*P;

VCA_ERR =norm(Mtrue-Mvca,'fro')/norm(Mtrue,'fro');



angles = Mtrue'*Mvsa./(repmat(sqrt(sum(Mtrue.^2)),p,1)'.*(repmat(sqrt(sum(Mvsa.^2)),p,1)));
P = zeros(p);
for i=1:p
    [dummy,j] = max(angles(i,:));
    P(j,i) = 1;
    angles(:,j) = -inf;
end
% permute colums
Mvsa = Mvsa*P;

MVSA_ERR =norm(Mtrue-Mvsa,'fro')/norm(Mtrue,'fro');

fprintf('\nERROR(mse): \n SISAL = %f\n MVSA = %f\n VCA = %f\n', SISAL_ERR,MVSA_ERR,VCA_ERR);


%--------------------------------------------------------------------------
%        Plot signatures
%-------------------------------------------------------------------------
% Choose signatures

leg_cell = cell(1);
H_3=figure;
hold on
clear p_H;

% plot signatures
p_H(1) = plot(1:L,(Up*Mtrue(:,1))','k');
leg_cell{end} = 'Mtrue';
p_H(2) =plot(1:L,(Up*Msisal(:,1))','r');
leg_cell{end+1} = 'Msisal';
p_H(3) =plot(1:L,(Up*Mvca(:,1))','b');
leg_cell{end+1} = 'Mvca';

for i=2:p
    plot(1:L,(Up*Mtrue(:,i))','k');
    plot(1:L,(Up*Msisal(:,i))','r');
    plot(1:L,(Up*Mvsa(:,i))','b');

end
legend(leg_cell)
title('First endmember')
xlabel('spectral band')

pos2 = get(H_2,'Position');
pos2(1)=50;
pos2(2)=1;
set(H_2,'Position', pos2)

pos3 = get(H_3,'Position');
pos3(1)=600;
pos3(2)=100+400;
set(H_3,'Position', pos3)


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %