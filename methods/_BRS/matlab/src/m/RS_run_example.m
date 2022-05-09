%RS_run_example Loads the dataset, creates output directories, runs a 
% testcase, and saves the results.
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

% make output folder
mkdir(folder_results);

if tracking
    param_alg.tracking_freq.trackObj = true;
    param_alg.tracking_freq.trackUk = true;
    param_alg.tracking_freq.obj = 1; %how often should objective function and  u/k be tracked?
    param_alg.tracking_freq.uk.iterations = [0:10:100, 200:100:2000, 3000:1000:5000];
    %[1:10,12:2:20,30:10:100,150:50:500,600:200:6000];
    global fftCount
    global fftCountSwitch
    fftCountSwitch = true;
end

if groundtruth_available
    load([dataset, '_ground_truth.mat']);
end
   
if isnumeric(param_alg.inertia)
    str_inertia = sprintf('%g', param_alg.inertia);
else
    str_inertia = param_alg.inertia;
end

% output arrays
if groundtruth_available
    SSIMVal = zeros(length(lambda_u), length(lambda_k));
    HPSIVal = zeros(length(lambda_u), length(lambda_k));
end

% make output folder for dataset and algorithm
name_testcase = strrep([dataset, '__', ...
    param_alg.algorithm, str_inertia], '.', '-');
folder_testcase = strrep([folder_results, filesep, ...
    name_testcase], '__', filesep);
mkdir(folder_testcase);

for ia = 1 : length(lambda_u)
    param_model.lambda_u = lambda_u(ia);
    
    for ib = 1 : length(lambda_k)
        param_model.lambda_k = lambda_k(ib);
        
        name_param = strrep(sprintf('lambda_u__%g__lambda_k_%g__gamma_%g__eta_%g',...
            param_model.lambda_u, param_model.lambda_k, ...
            param_model.gamma, param_model.eps), '.','-');
        
        filename = [name_testcase, '__', name_param];
        
        % make output folder
        folder_param = [folder_testcase, filesep, name_param];
        mkdir(folder_param);
        
        file = [folder_param, filesep, filename];
        
        if ~exist([file '.mat'], 'file')
            if tracking
                fftCount = 0;
            end
            
            [u_large, u, kernel, iter, data, Ax, objTrack, ukTrack] = ...
                RS_image_fusion(dataset, param_model, param_alg);
            
            save(file, 'u', 'kernel', 'iter', 'data', 'Ax');
            
            if tracking
                save(sprintf('%s_niter_%g_tracking', file, iter), ...
                    'objTrack', 'ukTrack');
            end
        else
            load(file);
            
            if tracking
                load(sprintf('%s_niter_%g_tracking', file, iter));
            end
        end
        
        file_iter = sprintf('%s_niter_%i', file, iter);
        
        if tracking
            fileID = fopen([file_iter, '_tracking.txt'],'w');
            fprintf(fileID,'%10s %10s %10s %10s %10s %10s %10s\n', ...
                'iter','fft', 'time', 'obj', 'data', 'Ru', 'Rk');
            for j = 1 : 10 : size(objTrack, 1)
                fprintf(fileID,'%10d %10d %10.4f %10.4e %10.4e %10.4e %10.4e\n', ...
                    objTrack(j, :));
            end
            fclose(fileID);
        end
        
        if groundtruth_available
            SSIMVal(ia, ib) = ssim(u, ground_truth);
            gt255 = 255 * min(ground_truth, 1);
            u255 = 255 * min(u, 1);
            HPSIVal(ia, ib) = HaarPSI(gt255, u255);
            
            imwrite(u - ground_truth + 0.5, [file_iter, '_image_error_.png'])
        end
        
        RS_save_image(u, [file_iter, '_image']);
        RS_save_image(Ax, [file_iter, '_data']);
        imwrite(Ax - data + 0.5, [file_iter, '_data_error.png']);
        RS_save_kernel(kernel, [file_iter, '_kernel']);
    end
end

if groundtruth_available
    name_stats = [name_testcase, ...
        '__lambda_u_', sprintf('%g_', lambda_u), ...
        '_lambda_k_', sprintf('%g_', lambda_k),...
        '_gamma_', sprintf('%g_', param_model.gamma)];
    name_stats = strrep([name_stats(1:end-1), '_stats'], '.', '-');
    filename_stats = [folder_testcase, filesep, name_stats];
    
    save(filename_stats, 'HPSIVal', 'SSIMVal');
    
    fileID = fopen([filename_stats, '.txt'], 'w');
    fprintf(fileID, '%10s %10s %10s %10s %10s %10s\n', 'lam_u', 'lam_k', 'i_lam_u', 'i_lam_k', 'ssim', 'hpsi');
    for ia = 1 : length(lambda_u)
        for ib = 1 : length(lambda_k)
            fprintf(fileID,'%10.4f %10.4f %10d %10d %10.4f %10.4f\n', ...
                lambda_u(ia), lambda_k(ib), ia, ib, ...
                100 * SSIMVal(ia,ib), 100 * HPSIVal(ia,ib));
        end
        fprintf(fileID,'\n');
    end
    fclose(fileID);
end

