clear all;
close all;

%% input parameters
testcases = {'trees1_shift_5px_disk', 'trees2_ch108_NW'};
lambda_us = {0.1, 1};
lambda_ks = {10, 10};
algorithms = {'PALM0', 'PALM0-1', 'PALM0-2', 'PALM0-5'};

xlab = 'iterations';

%% visualization parameters
cmx = 3.25; %width of output images in inches
cmy = 2.5; %height of output images in inches

fontsize = 13;
str_tickfontsize = '\fontsize{9}';
linewidth = 1.5;

legendStr = {'PALM', 'iPALM ($\alpha=0.1$)', 'iPALM ($\alpha=0.2$)', 'iPALM ($\alpha=0.5$)'};
linestyles = {'-','--',':','-.', '-','--',':','-.'};
folder = ['..', filesep, 'results', filesep, 'algorithm_comparison'];

xCol = 1; %1: iter, 2: ffts, 3: elapsed time
c_iter = 1;

n_algs = length(algorithms);
boundary = 20;

for i_testcase = 1:length(testcases)
    lambda_u = lambda_us{i_testcase};
    lambda_k = lambda_ks{i_testcase};

    % create directories
    testcase = testcases{i_testcase};

    param = strrep(sprintf('lambda_u__%g__lambda_k_%g__gamma_0-9995__eta_0-003', lambda_u, lambda_k), '.', '-');          

    folder_testcase = [folder, filesep, testcase]; 

    folder_summary = [folder_testcase, filesep, 'summary', filesep, strrep(sprintf('lambda_u_%g', lambda_u), '.', '-')];
    mkdir(folder_summary) 

    %% load objective values
    for i_alg = 1:n_algs
        alg = algorithms{i_alg};
        folder_result = [folder_testcase, filesep, alg, filesep, param];
        files = dir(folder_result);
        names = {files.name};
        for i = 1 : length(names)
            name = names{i};
            if length(name) > 12 && strcmp(name(end-11:end), 'tracking.mat')
                load([folder_result, filesep, name]);
                objTracks{i_alg} = objTrack;
                objTracks{i_alg}(1,3) = 1; %for log plot
                objTracks{i_alg}(1,2) = 1; %for log plot

                % save images and kernels
                for k = 1 : length(ukTrack)
                    u_inner = ukTrack{k}.u((1+boundary):(end-boundary),(1+boundary):(end-boundary));
                    RS_save_image(u_inner, [folder_result, filesep, testcase, '__', alg, '__', param, '__image_', sprintf('%0i', ukTrack{k}.n)])
                    RS_save_kernel(ukTrack{k}.k, [folder_result, filesep, testcase, '__', alg, '__', param, '__kernel_', sprintf('%0i', ukTrack{k}.n)])
                end                        
            end
        end
    end

    %% 'Legend.png'
    filename = 'Legend.png';

    hFig = figure(1);
    clf;

    hold on;
    hPlots = zeros(1,n_algs);
    for i_alg = 1:n_algs
        x = 1;
        y = 0;
        hPlots(i_alg) = plot(x, y, linestyles{i_alg}, 'LineWidth', linewidth);
    end

    legend(hPlots,[legendStr],'Location','NorthWest','Interpreter','latex');
    legend boxoff
    set(gca,'FontSize',fontsize);

    ax = gca;
    ax.Visible = 'off';
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 cmx cmy];
    print(hFig,[folder_summary, filesep, testcase, '__', param, '__', filename],'-dpng','-r300');

    %% 'ObjectiveDecay.png'
    ylab = 'relative objective $\frac{\Psi^t - \Psi^*}{\Psi^0 - \Psi^*}$';
    filename = 'ObjectiveDecay.png';
    c_obj = 4; %objective value
    yMinExp = -6;
    yMaxExp = 0;
    yticks = 10.^(yMinExp:2:yMaxExp);
    xticks = [0, 10, 100, 1000]+1;
    xlimits = [0, 5000];

    % set up labels and limits for x-Axis
%             xLabels = {'0','10','100','1000','5000'};%cellstr(num2str((xticks)'))';
    xLabels{1} = '0';
    for i = 2:length(xticks)
        xLabels{i} = RS_num2scientificLatex(xticks(i)-1);
    end

    for i = 1:length(xticks)
        xLabels{i} = [str_tickfontsize, xLabels{i}];
    end

    for i = 1:length(yticks)
        yLabels{i} = [str_tickfontsize, RS_num2scientificLatex(yticks(i))];
    end

    hFig = figure(c_obj);
    clf;

    hold on;
    hPlots = zeros(1,n_algs);
    for i_alg = 1:n_algs
        x = objTracks{i_alg}(:,xCol)+1;
        obj = objTracks{i_alg}(:, c_obj);
        y = (obj - obj(end))./(obj(1) - obj(end));
        hPlots(i_alg) = plot(x, y, linestyles{i_alg}, 'LineWidth', linewidth);
    end

    RS_print_format_and_save

    %% 'ObjectiveTracking.png'
    ylab = 'objective value $\Psi$';
    filename = 'ObjectiveTracking.png';
    c_obj = 4; %objective value

    hFig = figure(c_obj);
    clf;

    hold on;
    for i_alg = 1:length(algorithms)
        x = objTracks{i_alg}(:,xCol);
        y = objTracks{i_alg}(:,c_obj);
        hPlots(i_alg) = plot(x, y, linestyles{i_alg}, 'LineWidth', linewidth);
    end

    RS_print_ylabel            
    RS_print_format_and_save

    %% 'DataFidelityTracking.png'
    ylab = 'data fidelity $\mathcal{D}$';
    filename = 'DataFidelityTracking.png';
    c_obj = 5; %data fidelity

    hFig = figure(c_obj);
    clf;

    hold on;
    for i_alg = 1:length(algorithms)
        x = objTracks{i_alg}(:,xCol);
        y = objTracks{i_alg}(:,c_obj);
        hPlots(i_alg) = plot(x, y, linestyles{i_alg}, 'LineWidth', linewidth);
    end

    RS_print_ylabel
    RS_print_format_and_save

    %% 'RuTracking.png'
    ylab = 'image regularizer $\mathcal{R}_u$';
    filename = 'RuTracking.png';
    c_obj = 6; %image regularizer

    hFig = figure(c_obj);
    clf;

    hold on;
    for i_alg = 1:length(algorithms)
        x = objTracks{i_alg}(:,xCol);
        y = objTracks{i_alg}(:,c_obj);
        hPlots(i_alg) = plot(x, y, linestyles{i_alg}, 'LineWidth', linewidth);
    end

    RS_print_ylabel            
    RS_print_format_and_save

    %% 'RkTracking.png'
    ylab = 'kernel regularizer $\mathcal{R}_k$';
    filename = 'RkTracking.png';
    c_obj = 7; %kernel regularizer

    hFig = figure(c_obj);
    clf;

    hold on;
    for i_alg = 1:length(algorithms)
        x = objTracks{i_alg}(:,xCol);
        y = objTracks{i_alg}(:,c_obj);
        hPlots(i_alg) = plot(x, y, linestyles{i_alg}, 'LineWidth', linewidth);
    end

    RS_print_ylabel
    RS_print_format_and_save
end
