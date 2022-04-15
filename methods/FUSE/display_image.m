function output = display_image(imaff,colmap)

L = size(imaff,3);
for i=1:L
    m = min(min(imaff(:,:,i)));
    imaff(:,:,i) = imaff(:,:,i)-m;
    imaff(:,:,i) = imaff(:,:,i)/max(max(imaff(:,:,i)));
end

imaff = imadjust(imaff,stretchlim(imaff),[]);

imshow(imaff)
if nargin==2
    colormap(colmap)
end

output = 1;
