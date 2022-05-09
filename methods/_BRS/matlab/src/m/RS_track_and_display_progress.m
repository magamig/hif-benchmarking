function [objTrack, ukTrack] = RS_track_and_display_progress(objTrack, ukTrack, ...
    u, k, objective, trackingFreq, outputFreq, drawStateOfMinimization, ...
    ukFig, maxIter, n, A_u_k, Lu, Lk, diffu, diffk)
%RS_track_and_display_progress Tracks objective and iterates.
%
% Input:
%     objTrack [matrix]
%         information about objective function values, number of ffts, time, etc.       
%     ukTrack [cell]
%         various outputs for visualization and comparisons including
%         iterates of image and kernel
%     u [matrix]
%         current image iterate
%     k [matrix]
%         current kernel iterate      
%     objective [struct]
%         contains function handles for evaluation of the objective function, the data 
%         fidelity and the regularizers
%     trackingFreq [int]
%         tracking frequency        
%     outputFreq [int]
%         output frequency  
%     drawStateOfMinimization [logical]
%         plots during the optimization [true/false]         
%     ukFig [figure]
%         MATLAB figure         
%     maxIter [int]
%         number of iterations         
%     n [int]
%         number of current iteration         
%     A_u_k [matrix]
%         application of forward operator to image and kernel         
%     Lu, Lk [scalar]
%         Lipschitz constants         
%     diffu, diffk [scalar]
%         relative differences between the iterates   
%         
% Output:
%     objTrack [matrix]
%         information about objective function values, number of ffts, time, etc.
%     ukTrack [cell]
%         various outputs for visualization and comparisons including
%         iterates of image and kernel
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

        global fftCountSwitch
        global fftCount

        ifConverged = n < 0;
        currTimeElapsed = toc;
       
        if trackingFreq.trackObj
            ifTrackObj = ((mod(abs(n),trackingFreq.obj) == 0) || n == 0 || n == maxIter || ifConverged);
        else
            ifTrackObj = false;
        end
        if trackingFreq.trackUk
            ifTrackUk = (ismember(n,trackingFreq.uk.iterations) || n == 0 || n == maxIter || ifConverged);
        else
            ifTrackUk = false;
        end
        
        ifDisplayProgress = (outputFreq > 0 && ((mod(abs(n),outputFreq)==0) || n == 0 || n == maxIter || ifConverged));

        if ifTrackObj || ifTrackUk || ifDisplayProgress
            fftCountSwitch = false;

            data_fid = objective.data(A_u_k);
            Ru = objective.image(u);
            Rk = objective.kernel(k);
            total = data_fid + Ru + Rk;
            
            % trackObj
            if ifTrackObj
                if n == 0
                    currTimeElapsed = 0;
                end
                currState = [n, fftCount, currTimeElapsed,total, data_fid, Ru, Rk];
                objTrack = [objTrack; currState];
            end

            % trackUk
            if ifTrackUk
                if n == 0
                    currTimeElapsed = 0;
                end
                currStateuk.fftCount = fftCount;
                currStateuk.timeElapsed = currTimeElapsed;
                currStateuk.u = u;
                currStateuk.n = n;
                currStateuk.k = k;
                currStateuk.total = total;
                currStateuk.dataFidelity = data_fid;
                currStateuk.Ju = Ru;
                currStateuk.Jk = Rk;
                ukTrack{end+1} = currStateuk;
            end

            % displayProgress
            if  ifDisplayProgress
                if exist('Lu', 'var')
                    fprintf('Iter:%4d, obj:%4.4e(%3.3e|%3.3e|%3.3e), L:%2.2e|%2.2e, d:%2.2e|%2.2e, time:%2.2e\n',n, total, data_fid, Ru, Rk, Lu, Lk, diffu, diffk, currTimeElapsed);
                else
                    fprintf('Iter:%4d, obj:%2.2e(%2.2e|%2.2e|%2.2e), time:%2.2e\n',n, total, data_fid, Ru, Rk, currTimeElapsed);
                end

                if drawStateOfMinimization
                    figure(ukFig);
                    subplot(1,2,1);
                    imagesc(u); colorbar;
                    colormap gray;axis tight;axis equal;title(['u* after ',num2str(n),' iterations']);
                    drawnow;
                    subplot(1,2,2);
                    imagesc(k); colorbar;
                    colormap gray;axis tight;axis equal;title(['k* after ',num2str(n),' iterations']);
                    drawnow;
                end
            end

            fftCountSwitch = true;
        end
end 