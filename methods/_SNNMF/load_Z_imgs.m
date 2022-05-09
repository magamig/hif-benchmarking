function [ZIMGS scalar] = load_Z_imgs(filestem,roi,dsf)
% Load the High Resolution Hyperspectral Z *as images*
X1 = dsf*(roi(1)-1) + 1;
Y1 = dsf*(roi(3)-1) + 1;
X2 = dsf*roi(2);
Y2 = dsf*roi(4);

ZIMGS = zeros(Y2 - Y1 + 1, X2 - X1 + 1, 31);

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
    ZIMGS(:,:,band) = dcs;
end
scalar = max(max(max(ZIMGS)));
ZIMGS = ZIMGS/scalar;
end