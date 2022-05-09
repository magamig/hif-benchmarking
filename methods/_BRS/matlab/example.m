%Example file to run scripts to create the results from 
% 
% L. Bungert, D. Coomes, M. J. Ehrhardt, J. Rasch, R. Reisenhofer, C.-B. Schoenlieb, 
% Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation,
% Inverse Problems, 2017.
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

% initialize paths and some fixed parameters
RS_init;

% runs examples from the paper from src/scripts
RS_algorithm_comparison_trees1_shift_5px_disk
RS_algorithm_comparison_trees2_ch108_NW
RS_example_blind_v_nonblind
RS_example_groundtruth_disk
RS_example_groundtruth_gaussian
RS_example_trees1_NE_spectral_comparison
RS_example_trees2_ch108_NW
RS_example_TV_v_dTV
RS_example_urban_ch1_city
RS_example_urban_park_spectral_comparison
RS_example_comparison_gamma
RS_print_algorithm_comparison
