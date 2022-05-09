function [F] = create_F()
% Create the Spectral Response Transform Matrix F
    F = [2  1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  2  6 11 17 21 22 21 20 20 19 19 18 18 17 17;...
         1  1  1  1  1  1  2  4  6  8 11 16 19 21 20 18 16 14 11  7  5  3  2  2  1  1  2  2  2  2  2;...
         7 10 15 19 25 29 30 29 27 22 16  9  2  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1];
     
    for band = 1:3
        div = sum(F(band,:));
        for i = 1:31
            F(band,i) = F(band,i)/div;
        end
    end
end