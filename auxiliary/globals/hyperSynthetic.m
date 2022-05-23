function [ hyper, multi ] = hyperSynthetic( truth, srf, downscale, h )
%Create synthetic hyperspectral data (hyper, multi)
% CL 2015

% downscale = 32;

if nargin==4
    if ~mod(size(truth,2)/h / downscale,1) && ~mod(h /downscale,1)
        S = hyperSpatialDown(h, size(truth,2)/h ,downscale);
    else
        error('The dimensions for spatial downsampling are not valid.')
    end
else
    S = hyperSpatialDown(sqrt(size(truth,2)),sqrt(size(truth,2)),downscale);
end

hyper = truth*S;
multi = srf*truth;


%visualize images
% figure;
% hyperImshow(multi)
% figure;
% hyperImshow(hyper)


end

