           
yMin = min(objTracks{1}(:,c_obj));
yMax = max(objTracks{1}(:,c_obj));
for i_alg = 2:length(algorithms)
    yMin = min(min(objTracks{i_alg}(:,c_obj)),yMin);
    yMax = max(max(objTracks{i_alg}(:,c_obj)),yMax);
end

if yMin > 1e-6
    k = floor(log10(yMin));
    yMin = 10^k * floor(yMin / 10^k);
    k = floor(log10(yMax));
    yMax = 10^k * ceil(yMax / 10^k);
    yMean = sqrt(yMin*yMax);
    k = floor(log10(yMean));
    yMean = 10^k * floor(yMean / 10^k);
    
    if yMean > yMin
        yticks = [yMin, yMean, yMax];
    else
        yticks = [yMin, yMax];
    end
    
    yLabels = {};
    for i = 1:length(yticks)
        yLabels = [yLabels,RS_num2scientificLatex(yticks(i))];
    end
    
    for i = 1:length(yLabels)
        yLabels{i} = [str_tickfontsize, yLabels{i}];
    end
    
    ylim([yMin,yMax]);
    set(gca,'YTick',yticks);
    set(gca,'YTickLabels',yLabels);
end