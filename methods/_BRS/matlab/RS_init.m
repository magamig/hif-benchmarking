%RS_init Initializes important model and algorithm parameters.
%         
% See also:
%
% -------------------------------------------------------------------------
% Copyright 2017, L. Bungert, D. Coomes, M. J. Ehrhardt, J. Rasch, 
% R. Reisenhofer, C.-B. Schoenlieb
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------

addpath(genpath(pwd)); % add subfolders

% dataset 
groundtruth_available = false;

% algorithm parameters
param_alg.niter = 2000;
param_alg.blind = true; % blind: update both image and kernel
param_alg.algorithm = 'PALM'; % options: PALM, PAM
param_alg.inertia = 0;
tracking = false;

% model parameters
param_model.eps = 0.003; % edge parameter
param_model.gamma = 0.9995; % scalar factor of vectorfield: 0 <= gamma <= 1

% output parameters
param_alg.draw_iterates = false;
param_alg.verbose_freq = 10;

