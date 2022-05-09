function [similarity,similarityMaps,weightMaps] = HaarPSI(imgRef,imgDist,preprocessWithSubsampling)
%HaarPSI Computes the Haar wavelet-based perceptual similarity index of two
%images.
%
%Please make sure that grayscale and color values are given in the [0,255]
%interval! If this is not the case, the HaarPSI cannot be computed
%correctly.
%
%Usage (optional parameters in <>-brackets):
%
% [similarity,<similarityMaps>,<weightMaps>] = HaarPSI(imgRef, imgDist, <preprocessWithSubsampling>);
%
%
%
%Input:
%
%                       imgRef: RGB or grayscale image with values ranging from 0
%                               to 255.
%                      imgDist: RGB or grayscale image with values ranging from 0
%                               to 255.
%    preprocessWithSubsampling: <optional> If 0, the preprocssing step to acommodate for the 
%                               viewing distance in psychophysical experimentes is omitted.
%                  
%
%Output:
%
%                   similarity: The Haar wavelet-based perceptual similarity index, measured
%                               in the interval [0,1].
%               similarityMaps: <optional> Maps of horizontal and vertical local similarities.
%                               For RGB images, this variable also includes
%                               a similarity map with respect to the two
%                               color channesl in the YIQ space.
%                   weightMaps: <optional> Weight maps measuring the importance of
%                               the local similarities in similarityMaps.
%                               
%
%Example:
%
% imgRef = double(imread('peppers.png'));
% imgDist = imgRef + randi([-20,20],size(imgRef));
% imgDist = min(max(imgDist,0),255);
% similarity = HaarPSI(imgRef,imgDist);
%
%Reference: 
%
% R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand: 'A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment', 2017.

    if nargin < 3
        preprocessWithSubsampling = 1;
    end
    colorImage = size(imgRef,3) == 3;    
        
    imgRef = double(imgRef);
    imgDist = double(imgDist);
    
    %% initialization and preprocessing   
    %constants
    C = 30;
    alpha = 4.2;
    %transform to YIQ colorspace
    if colorImage        
        imgRefY = 0.299 * (imgRef(:,:,1)) + 0.587 * (imgRef(:,:,2)) + 0.114 * (imgRef(:,:,3));
        imgDistY = 0.299 * (imgDist(:,:,1)) + 0.587 * (imgDist(:,:,2)) + 0.114 * (imgDist(:,:,3));
        imgRefI = 0.596 * (imgRef(:,:,1)) - 0.274 * (imgRef(:,:,2)) - 0.322 * (imgRef(:,:,3));
        imgDistI = 0.596 * (imgDist(:,:,1)) - 0.274 * (imgDist(:,:,2)) - 0.322 * (imgDist(:,:,3));
        imgRefQ = 0.211 * (imgRef(:,:,1)) - 0.523 * (imgRef(:,:,2)) + 0.312 * (imgRef(:,:,3));
        imgDistQ = 0.211 * (imgDist(:,:,1)) - 0.523 * (imgDist(:,:,2)) + 0.312 * (imgDist(:,:,3));
    else
        imgRefY = double(imgRef);
        imgDistY = double(imgDist);
    end
       
    %% subsampling    
    if preprocessWithSubsampling
        imgRefY = HaarPSISubsample(imgRefY);
        imgDistY = HaarPSISubsample(imgDistY);    
        if colorImage
            imgRefQ = HaarPSISubsample(imgRefQ);
            imgDistQ = HaarPSISubsample(imgDistQ);
            imgRefI = HaarPSISubsample(imgRefI);
            imgDistI = HaarPSISubsample(imgDistI);
        end 
    end
    
    %% pre-allocate variables
    if colorImage
        localSimilarities = zeros([size(imgRefY),3]);
        weights = zeros([size(imgRefY),3]);  
    else
        localSimilarities = zeros([size(imgRefY),2]);
        weights = zeros([size(imgRefY),2]);  
    end    
    
    %% Haar wavelet decomposition
    nScales = 3;
    coeffsRefY = HaarPSIDec(imgRefY,nScales);
    coeffsDistY = HaarPSIDec(imgDistY,nScales);    
    if colorImage
        coeffsRefQ = abs(conv2(imgRefQ,ones(2,2)/4,'same'));
        coeffsDistQ = abs(conv2(imgDistQ,ones(2,2)/4,'same'));
        coeffsRefI = abs(conv2(imgRefI,ones(2,2)/4,'same'));
        coeffsDistI = abs(conv2(imgDistI,ones(2,2)/4,'same'));
    end
    
    %% compute weights and similarity for each orientation
    for ori = 1:2        
        weights(:,:,ori) = max(abs(coeffsRefY(:,:,3 + (ori-1)*nScales)), abs(coeffsDistY(:,:,3 + (ori-1)*nScales)));
        coeffsRefYMag = abs(coeffsRefY(:,:,(1:2) + (ori-1)*nScales));
        coeffsDistYMag = abs(coeffsDistY(:,:,(1:2) + (ori-1)*nScales));
        localSimilarities(:,:,ori) = sum(((2*coeffsRefYMag.*coeffsDistYMag + C)./(coeffsRefYMag.^2 + coeffsDistYMag.^2 + C)),3)/2;    
    end
    
    %% compute similarities for color channels
    if colorImage         
        similarityI = ((2*coeffsRefI.*coeffsDistI + C) ./(coeffsRefI.^2 + coeffsDistI.^2 + C));
        similarityQ = ((2*coeffsRefQ.*coeffsDistQ + C) ./(coeffsRefQ.^2 + coeffsDistQ.^2 + C));
        localSimilarities(:,:,3) = ((similarityI)+(similarityQ))/2;
        weights(:,:,3) = (weights(:,:,1) + weights(:,:,2))/2;
    end
    
    %% compute final score
    similarity = HaarPSILogInv(sum((HaarPSILog(localSimilarities(:),alpha)).*weights(:))/sum(weights(:)),alpha)^2;
    
    %% output maps
    if nargout > 1
        similarityMaps = localSimilarities;
    end
    if nargout > 2
        weightMaps = weights;
    end
end

function coeffs = HaarPSIDec(X,nScales)
    coeffs = zeros([size(X),2*nScales]);    
    for k = 1:nScales
        haarFilter = 2^(-k)*ones(2^k,2^k);
        haarFilter(1:(end/2),:) = -haarFilter(1:(end/2),:);
        coeffs(:,:,k) = conv2(X,haarFilter,'same');
        coeffs(:,:,k + nScales) = conv2(X,haarFilter','same');
    end   
end

function imgSubsampled = HaarPSISubsample(img)
    imgSubsampled = conv2(img, ones(2,2)/4,'same');        
    imgSubsampled = imgSubsampled(1:2:end,1:2:end);
end

function val = HaarPSILog(x,alpha)
    val = 1./(1 + exp(-alpha.*(x)));
end

function val = HaarPSILogInv(x,alpha)
    val = log(x./(1-x))./alpha;
end

%  Written by Rafael Reisenhofer
%  Built on 08/05/2017

