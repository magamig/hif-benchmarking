function [sam] = GetSAMofHSI( oI,I,row,col )
%   Calculate the average SAM score of the two HSIs
% oI: the original image of size [row,col, band]
% I: the recovery image of size [row, col, band]
%

oI3d = CovertTo3D( oI,row,col );
I3d = CovertTo3D( I,row,col );
sam = SpectAngMapper(oI3d,I3d);

end

