function [down_im] = downsample1(im, scale)
new_pixel_size = scale;
a = size(im,1)/new_pixel_size;
down_im = zeros(a,a,size(im,3));
m = zeros(a, size(im,1));
s = 1:new_pixel_size:size(im,1);
e = new_pixel_size:new_pixel_size:size(im,1);

for l = 1:size(im,3)
    ima(:,:) = im(:,:,l);
    temp1 = [];
    for i = 1:a
        t = sum(ima(s(i):e(i), :));
        temp1 = [temp1; t];
    end 
    temp2 = [];
    for i = 1:a 
        t = sum(temp1(:,s(i):e(i)),2);
        temp2 = [temp2 t];
    end
    temp2 = temp2./repmat(new_pixel_size^2,size(temp2,1),size(temp2,1));
    down_im(:,:,l) = temp2;
end
    
end

