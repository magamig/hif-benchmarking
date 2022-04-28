function [y,x,noise,sigma,outliers] = spectMixGen(M,samp_size,varargin)
%%
% Usage:
% [y,x,noise,outliers] = spectMixGen(M,samp_size,varargin)
%
% This function generates a simulated spectral data set
%
%     y = M*x+noise
%
% where M is the mixing matrix containing the endmembers, x is the
% the fractions (sources) of each enmember at each pixel, and noise is a
% Gaussian independent (band and pixel - wise) additive perturbation.
% 
%
% 
% Author: Jose Bioucas-Dias, January, 2009.
% 
% Please check for the latest version of the code and papers at
% www.lx.it.pt/~bioucas/SUNMIX
%
% 
%
%
%  =====================================================================
%  ===== Required inputs ===============================================
%
%  M: [Lxp] mixing matrix containing the p endmembers of size L 
%     
%  samp_size: number of spectral vectors to be generated
%
%  
%  ===== Optional inputs =============
% 
%  
%  'Source_pdf' = Source densities  {'uniform', 'Diri_id', 'Diri_mix'}
%                'uniform'   -> uniform over the simplex
%                'Diri_id'   -> Direchlet with equal parameters
%                'Diri_mix'  -> Mixture of Direchlet densities
%
%                Default = 'uniform'
%
%  'pdf_pars'  = pdf parameters 
%                if ('Source_pdf' == 'Diri_id') 
%                      pdf_pars = a > 0 ; p(x) \sim D(a,...a) 
%                if ('Source_pdf' == 'Diri_mix')
%                      pdf_pars = A; [m,p+1] ;
%                      Each line of A conatins the parameters of a 
%                      Direchlet mode. 
%                      A(i,1)    -> weights of mode i  (0 < A(i,1)<= 1, sum(A(:,1)) = 1)
%                      A(i,2:p)  -> Dirichelet parameters of mode i  
%                                   (A(i,j)> 0, j>=2)
%                Default = 1 (<==> uniform);
%
%   'pure_pixels'  = include pure pixels in the data set  {'yes', 'no'}
%                    
%                   Default = 'no' 
%
%    'max_purity' = vector containing maximum purities. Is a scalat is
%                   passed, it is interpreted as a vector with components
%                   equal to the passed vscalar.
%
%                 Default = [1 , ..., 1]
%
%    'am_modulation' = multiplies the sources by a random scale factor
%                      uniformely distributed in an the interval.
%
%                   Default = [1 1]  % <==> no amplitude modulation
%
%    'sig_variability' = multiply each component of each source with a 
%                        a random scale factor uniformely distributed in an the interval
%
%                   Default = [1 1]  % <==> no asignature variability.
%                           
%
%    'no_outliers' = number of vector outside the simplex (sum(x) = 1, but some x_i > 1 )
%
%                    Default = 0;
%
%    'violation_extremes' = [min(x), max(x)] in case of  no_outliers > 0
%
%                           Default = [1 1.2];
%    
%    'snr' = signal-to-noise ratio in dBs    
% 
%           Default = 40 dBs
%                   
%
%    'noise_shape' = shape of the Gaussian variace along the bands
%                    {'uniform', 'gaussian', 'step', 'rectangular'}
%                     'uniform'     -> equal variance
%                     'gaussian'    -> Gaussian shaped variance centered at
%                                      b and with spread eta:
%                                      1+ beta*exp(-[(i-b)/eta]^2/2)
%                     'step'        -> step  and amplitudes (1,beta) centered
%                                      at b 
%                     'rectangular' ->  1+ beta*rect(i-b)/eta
% 
%                     Default = 'uniform';
%
%    'noise_pars'  = noise parameters [beta  b eta]
%
%                    Default = [0 1 1];  <==> 'uniform'
%
%    
%
% ===================================================  
% ============ Outputs ==============================
%
% [y,x,noise,outliers]
%   y = [Lxsamp_size] data set
%
%   x =  [pxsamp_size] fractional abundances
%   
%   noise = [Lxsamp_size] additive noise 
%
%   sigma = [Lx1]  vector with the noise standard deviations along the L bands
%
%   outliers = [pxno_outliers] source oultliers
%
% ========================================================
%
% NOTE: The order in which the degradation mechanisms are input is
% irrelevant. However, since the degradation mechanism are not comutative,
% the  implemented order is relevent and it is the following:
%
%    1)  generate sources 
%    2)  enforces max purity
%    3)  include pure pixels 
%    4)  include outliers 
%    5)  amplitude modulation 
%    6)  signatute variability
%    7)  generate noise
%    
% ===================================================  
% ============ Call examples ==============================
%
%
%  [y,x,noise,outliers] = spectMixGen(M,samp_size)
%
%  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', 'Diri_id', 'pdf_pars', 5)
%
%  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', 'Diri_mix', 'pdf_pars', [1, 0.1 1 2 3])
%
%  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', ...
%                                                    'Diri_mix', ...'pdf_pars', [0.2, 0.1 1 2 3,
%                                                                                0.8, 2   3 4 5])
%
%  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', 'Diri_id',
%                     'snr', 20, 'pdf_pars', 5,  'max_purity', [0.8 1 1 1],  'noise_shape',
%                      'gaussian', 'noise_pars', [2,50,20])
%
%  [y,x,noise,outliers] = spectMixGen(M,samp_size)
%







%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
     error('Wrong number of required parameters');
end

% endmember matrix size 
[L,p] = size(M); %((L-> number of bands, p -> number of endmembers)


%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------

source_pdf   = 'uniform';
pdf_pars = 1;
pure_pixels = 'no';
max_purity = ones(1,p);
no_outliers = 0;
violation_extremes = [1 1.2];
am_modulation = [1 1];
sig_variability = [1 1];
snr = 40;     % 40 dBs
noise_shape = 'uniform';
noise_pars = [0 1 1];


%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'SOURCE_PDF'
       source_pdf = varargin{i+1};
     case 'PDF_PARS'
       pdf_pars = varargin{i+1};
     case 'PURE_PIXELS'
       pure_pixels = varargin{i+1};
     case 'MAX_PURITY'
       max_purity = varargin{i+1};
     case 'NO_OUTLIERS'
       no_outliers = varargin{i+1};
     case 'VIOLATION_EXTREMES'
       violation_extremes = varargin{i+1}; 
     case 'AM_MODULATION'
       am_modulation = varargin{i+1}; 
     case 'SIG_VARIABILITY'
       sig_variability = varargin{i+1}; 
     case 'SNR'
       snr = varargin{i+1};
     case 'NOISE_SHAPE'       
       noise_shape = varargin{i+1};
     case 'NOISE_PARS'
       noise_pars = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%

%% generate sources 

% check for validity
if  strcmp(source_pdf, 'Diri_mix')
   [no_modes,cols] = size(pdf_pars); 
   if cols ~= (p+1)
        error('Wrong pdf parameters');
   elseif (sum(pdf_pars(:,1)) ~=1) || (sum(pdf_pars(:,1) < 0) > 0)
        error('Wrong pdf parameters -> mixing weights  do not define a probability'); 
   end
else
    no_modes = 1;
end
    
% take the density as a mixture in all cases (weights - MOD weights; pdf_pars(i,:) - MOD_i parameters')
 switch  source_pdf
    case 'uniform'
        pdf_pars = ones(1,p);
        weights = 1;
    case 'Diri_id'
           pdf_pars = pdf_pars(1)*ones(1,p);
           weights = 1;
    case'Diri_mix'
        weights = pdf_pars(:,1);
        pdf_pars = pdf_pars(:,2:end);  
 end
        

 % determine  the size of each lenght
 mode_length = round(weights*samp_size);
 
 % correct for rounding erros
 mode_length(no_modes) = mode_length(no_modes) + samp_size - sum(mode_length) ; 
 
 x = [];
 for i=1:no_modes
    x = [x dirichlet(pdf_pars(i,:),mode_length(i))']; 
 end
 
 % do a random permutation of columns (not necessary)
 x = x(:,randperm(samp_size));
 

%% enforces max purity

% if max_purity is a scalar, convert it into a vector
if length(max_purity) == 1
    max_purity = max_purity * ones(1,p);
end

% check for validity
if (sum(max_purity < 0 ) + sum(max_purity > 1)) > 0
      error('Purity must be in (0,1)'); 
elseif sum(max_purity) <1 
    error('sum (max_purity) can not be less than 1');
end
    
% ensure that is a line vector
max_purity = reshape(max_purity,1,p);

if sum(max_purity) < p
    % threshold sources
    x_max = repmat(max_purity',1,samp_size);
    % build a  mask
    mask = x <= x_max;
    % threshold sources
    x_th = x.*mask + (1-mask).*x_max;
    % sources excess
    x_excess = x-x_th;
    % total of excess per pixel
    x_acum = sum(x_excess);
    % slack per peixel
    slack = x_max - x_th;
    % redistribute total acumulated proportionali to the individual slack
    x = x_th+slack./repmat(sum(slack),p,1).*repmat(x_acum,p,1);
    clear x_max mask slack x_th;
end


%% include pure pixels (at the end)
if strcmp(pure_pixels,'yes')
     x = [x(:,p+1:end) eye(p)];
end
 
%% include outliers (at the begining)
% we simple pick up the first no_outliers pixels and forces one of its 
% fractions  to be in the interval violation_extremes an to sum 1
spread = violation_extremes(2)-violation_extremes(1);
for i=1:no_outliers
    index= randperm(p);
    x(index(1),i) = violation_extremes(1) + spread*rand(1);
    aux = rand(p-1,1);
    aux = aux./repmat(sum(aux),p-1,1)-x(index(1),i)/(p-1);
    x(index(2:p),i) = aux;
  end

outliers = x(:,1:no_outliers);

%% amplitude modulation 
am_length = am_modulation(2)- am_modulation(1);
if ((am_length) > 0 ) && ...   % apply amplitude modulation only if the 
        am_modulation(1) >= 0                            % the interval in positive
    x = x.*repmat(am_modulation(1)+ am_length*rand(1,size(x,2)),p,1);                                                  
end

%% signature  variability 
sig_length = sig_variability(2)- sig_variability(1);
if ((sig_length) > 0 ) && ...   % apply amplitude modulation only if the 
        sig_variability(1) >= 0                            % the interval in positive
    x = x.*(sig_variability(1)+ sig_length*rand(size(x)));                                                  
end



%% generate noise
 
% generate noise shape (sum of variances is one)
beta = noise_pars(1); 
b = noise_pars(2); 
eta = noise_pars(3); 

xx = (1:L)';

switch  noise_shape     
    case 'uniform' 
       sigma = ones(L,1);  
    case'gaussian'
       sigma = 1+ beta*exp(-[(xx-b)/eta].^2/2);
    case'step'
       sigma = 1+ beta*(xx >= b);
    case'rectangular'
       sigma = 1+ beta*(abs((xx-b)/eta) < 1);
 
end

% normalize
sigma = sigma/sqrt(sum(sigma.^2));
 
% compute mean variance
% generate data without noise
y=M*x;

sigma_mean = sqrt((sum(y(:).^2)/samp_size)/(10^(snr/10)));

sigma = sigma_mean*sigma;

noise = diag(sigma)*randn(size(y));

y = y+noise;
return




