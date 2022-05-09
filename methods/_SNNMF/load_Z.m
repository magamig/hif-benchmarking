function [Z] = load_Z(filestem, roi, dsf)
% Load the High Resolution Hyperspectral Z
X1 = dsf*(roi(1)-1) + 1;
Y1 = dsf*(roi(3)-1) + 1;
X2 = dsf*roi(2);
Y2 = dsf*roi(4);

X0F = zeros(31,(Y2 - Y1 + 1)*(X2 - X1 + 1));

for band = 1:31
    filesffx = '.png';
    
    if band < 10
        prefix = 'ms_0';
    else
        prefix = 'ms_';
    end
    
    number = strcat(prefix, int2str(band));
    filename = strcat(filestem, number, filesffx);

    cur = imread(filename);
    dcs = cur(Y1:Y2,X1:X2);
    
    for row = 1:(Y2 - Y1 + 1)
        for col = 1:(X2 - X1 + 1)
            X0F(band,col + (X2 - X1 + 1)*(row-1)) = dcs(row,col);
        end
    end
end

Z = X0F/max(max(X0F));
end