function img = hyperConvert3d(img,h,w)

if ndims(img)~=2
    warning('Image is not 2D.')
    return
end

[b,N] = size(img);

if nargin==1
    h = sqrt(N);
    w = h;
    if round(h)~=h
        while mod(N/h,1)>0
            h = floor(h)-1;
            w = N/h;
        end
    end
elseif nargin==2
    w = N/h;
end

img = reshape(img',[h,w,b]);
