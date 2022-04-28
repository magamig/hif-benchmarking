function HSI = gaussian_down_sample(data,w,mask)
%------------------------------------------------------------------
% This function downsamples HS image with Gaussian point spread function.
% 
%
% HSI = gaussian_down_sample(data,w)
%
% INPUT
%       data            : input HS image (xdata,ydata,band)
%       w               : difference of ground sampling distance (FWHM = w)
%       mask(optional)  : masking map (xdata,ydata)
%
% OUTPUT
%       HSI             : downsampled HS image (band, xdata/w, ydata/w)
%
% AUTHOR
% Naoto Yokoya, University of Tokyo
% Email: yokoya@sal.rcast.u-tokyo.ac.jp
%
%------------------------------------------------------------------

if nargin == 3
    masking = 2;
else
    masking = 1;
end

[xdata, ydata, band] = size(data);
hx = floor(xdata/w); 
hy = floor(ydata/w);
HSI = zeros(hx, hy, band);
sig = w/2.35482;

switch masking
    case 1
if mod(w,2)==0
    H1 = fspecial('gaussian',[w w],sig);
    H2 = fspecial('gaussian',[w*2 w*2],sig);
    for x = 1:hx
        for y = 1:hy
            if x==1 || x==hx || y==1 || y==hy
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w:x*w,1+(y-1)*w:y*w,:)).*repmat(H1,[1 1 band]),1),2);
            else
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w-w/2:x*w+w/2,1+(y-1)*w-w/2:y*w+w/2,:)).*repmat(H2,[1 1 band]),1),2);
            end
        end
    end
else
    H1 = fspecial('gaussian',[w w],sig);
    H2 = fspecial('gaussian',[w*2-1 w*2-1],sig);
    for x = 1:hx
        for y = 1:hy
            if x==1 || x==hx || y==1 || y==hy
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w:x*w,1+(y-1)*w:y*w,:)).*repmat(H1,[1 1 band]),1),2);
            else
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w-(w-1)/2:x*w+(w-1)/2,1+(y-1)*w-(w-1)/2:y*w+(w-1)/2,:)).*repmat(H2,[1 1 band]),1),2);
            end
        end
    end
end

    case 2
if mod(w,2)==0
    H1 = fspecial('gaussian',[w w],sig);
    H2 = fspecial('gaussian',[w*2 w*2],sig);
    for x = 1:hx
        for y = 1:hy
            mask_tmp = mask(1+(x-1)*w:x*w,1+(y-1)*w:y*w);
            if sum(mask_tmp(:)) == 0
                if x==1 || x==hx || y==1 || y==hy
                    HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w:x*w,1+(y-1)*w:y*w,:)).*repmat(H1,[1 1 band]),1),2);
                else
                    HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w-w/2:x*w+w/2,1+(y-1)*w-w/2:y*w+w/2,:)).*repmat(H2,[1 1 band]),1),2);
                end
            end
        end
    end
else
    H1 = fspecial('gaussian',[w w],sig);
    H2 = fspecial('gaussian',[w*2-1 w*2-1],sig);
    for x = 1:hx
        for y = 1:hy
            mask_tmp = mask(1+(x-1)*w:x*w,1+(y-1)*w:y*w);
            if sum(mask_tmp(:)) == 0
                if x==1 || x==hx || y==1 || y==hy
                    HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w:x*w,1+(y-1)*w:y*w,:)).*repmat(H1,[1 1 band]),1),2);
                else
                    HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w-(w-1)/2:x*w+(w-1)/2,1+(y-1)*w-(w-1)/2:y*w+(w-1)/2,:)).*repmat(H2,[1 1 band]),1),2);
                end
            end
        end
    end
end

end