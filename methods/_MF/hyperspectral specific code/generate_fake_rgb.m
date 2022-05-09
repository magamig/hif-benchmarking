function dataRGB = generate_fake_rgb(dataHS,P_rgb)

[w,h,l] = size(dataHS);
dataRGB = zeros(w,h,3);

for i = 1:w,
    for j = 1:h,
        dataRGB(i,j,:) = P_rgb' * reshape( dataHS(i,j,:), l, 1);
    end
end