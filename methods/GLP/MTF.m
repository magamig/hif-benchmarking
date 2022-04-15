%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF filters the image I_HS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the HyperSpectral (HS) sensor. 
% 
% Interface:
%           I_Filtered = MTF(I_HS,sensor,ratio)
%
% Inputs:
%           I_HS:           HS image;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
%           ratio:          Scale ratio between MS and PAN.
%
% Outputs:
%           I_Filtered:     Output filtered HS image.
% 
% References:
%           [Aiazzi06]   B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,”
%                        Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
%           [Lee10]      J. Lee and C. Lee, “Fast and efficient panchromatic sharpening,” IEEE Transactions on Geoscience and Remote Sensing, vol. 48, no. 1,
%                        pp. 155–163, January 2010.
%           [Vivone14]   G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                        IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Filtered = MTF(I_HS,sensor,ratio)

switch sensor
    case 'none'
        GNyq = 0.3 .* ones(1,size(I_HS,3)); % MTF to be defined for each sensor (default values).
end

%%% MTF

N = 31;
I_MS_LP = zeros(size(I_HS));
nBands = size(I_HS,3);
fcut = 1/ratio;
   
for ii = 1 : nBands
    alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    h = fwind1(Hd,kaiser(N));
    I_MS_LP(:,:,ii) = imfilter(I_HS(:,:,ii),real(h),'replicate');
end

I_Filtered= double(I_MS_LP);

end