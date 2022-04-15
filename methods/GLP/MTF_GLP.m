%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF_GLP fuses the upsampled HyperSpectral (HS) and PANchromatic (PAN) images by 
%           exploiting the Modulation Transfer Function - Generalized Laplacian Pyramid (MTF-GLP) algorithm. 
% 
% Interface:
%           I_Fus_MTF_GLP = MTF_GLP(HS,PAN,sensor,ratio)
%
% Inputs:
%           HS:                 HS image;
%           PAN:                PAN image;
%           sensor:             String for type of sensor (e.g. 'CHRIS-Proba', etc.);
%           ratio:              Scale ratio between HS and PAN;
%           th_value:           Flag to force values in the radiometric range;
%           L:                  Radiometric resolution.
%
% Outputs:
%           I_Fus_MTF_GLP:      MTF_GLP pansharpened image.
% 
% References:
%           [Aiazzi02]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on
%                       oversampled multiresolution analysis,?IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300?312, October
%                       2002.
%           [Aiazzi06]  B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,?
%                       Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591?96, May 2006.
%           [Vivone14a] G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral
%                       image pansharpening,?IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930?34, May 2014.
%           [Vivone14b] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_MTF_GLP = MTF_GLP(HS,PAN,sensor,ratio,th_value,L,posi)

imageHR = double(PAN);
HS = double(HS);

%%%%%%%% Upsampling
HSU = interp23tapGeneral(HS,ratio,posi);
%HSU = upsampling(HS,ratio);

%%% Equalization
imageHR = repmat(imageHR,[1 1 size(HSU,3)]);

% Different from the original MTF_GLP code, the following part has been
% commented out for hypersharpening since it does not improve the
% reconstruction performance.
% for ii = 1 : size(HSU,3)    
%     imageHR(:,:,ii) = (imageHR(:,:,ii) - mean2(imageHR(:,:,ii))).*(std2(HSU(:,:,ii))./std2(imageHR(:,:,ii))) + mean2(HSU(:,:,ii));
% end

PAN_LP = zeros(size(HSU));
nBands = size(HSU,3);
        
switch sensor
    case 'none'
        GNyq = 0.3 .* ones(1,size(HSU,3)); % MTF to be defined for each sensor (default values).
        
        %%% MTF
        N = 31;
        fcut = 1/ratio;
        PSF_G = zeros(N,N,nBands);

        for ii = 1 : nBands
            alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(ii))));
            H = fspecial('gaussian', N, alpha);
            Hd = H./max(H(:));
            h = fwind1(Hd,kaiser(N));
            PSF_G(:,:,ii) = real(h);
            PAN_LP(:,:,ii) = imfilter(imageHR(:,:,ii),real(h),'replicate');
            t = PAN_LP(:,:,ii);
            %start_pos(1)=1; % The starting point of downsampling
            %start_pos(2)=1; % The starting point of downsampling
            start_pos = posi;
            t = t(start_pos(1):ratio:end, start_pos(2):ratio:end,:);
            PAN_LP(:,:,ii) = interp23tapGeneral(t,ratio,posi);
        end
    case 'GaussKernel'
        %size_kernel=[9 9];
        %sig = (1/(2*(2.7725887)/ratio^2))^0.5;
        %start_pos(1)=1; % The starting point of downsampling
        %start_pos(2)=1; % The starting point of downsampling
        %t = conv_downsample(imageHR,ratio,size_kernel,sig,posi);
        t = gaussian_down_sample(imageHR,ratio);
        for ii = 1 : nBands
            PAN_LP(:,:,ii) = interp23tapGeneral(t(:,:,ii),ratio,posi);
        end
       
end

PAN_LP = double(PAN_LP);
C = cov(HSU(:),PAN_LP(:));
scaling = C(1,2)/C(2,2);
I_Fus_MTF_GLP = HSU + scaling*(imageHR - PAN_LP);

if th_value
    I_Fus_MTF_GLP(I_Fus_MTF_GLP < 0) = 0;
    I_Fus_MTF_GLP(I_Fus_MTF_GLP > 2^L) = 2^L;
end

end