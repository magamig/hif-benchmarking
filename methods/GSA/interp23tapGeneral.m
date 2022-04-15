function I_Interpolated = interp23tapGeneral(I_Interpolated,ratio)

L = 44; % tap

[r,c,b] = size(I_Interpolated);

BaseCoeff = ratio.*fir1(L,1./ratio);

I1LRU = zeros(ratio.*r, ratio.*c, b);    
% I1LRU(floor(ratio/2)+1:ratio:end,floor(ratio/2)+1:ratio:end,:) = I_Interpolated;
I1LRU(1:ratio:end,1:ratio:end,:) = I_Interpolated;

for ii = 1 : b
    t = I1LRU(:,:,ii); 
    t = imfilter(t',BaseCoeff,'circular'); 
    I1LRU(:,:,ii) = imfilter(t',BaseCoeff,'circular'); 
end

I_Interpolated = I1LRU;

end