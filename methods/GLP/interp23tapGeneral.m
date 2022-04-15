function I_Interpolated = interp23tapGeneral(I_Interpolated,ratio,start_pos)

L = 44; % tap

[r,c,b] = size(I_Interpolated);

BaseCoeff = ratio.*fir1(L,1./ratio);

I1LRU = zeros(ratio.*r, ratio.*c, b);    
% I1LRU(floor(ratio/2)+1:ratio:end,floor(ratio/2)+1:ratio:end,:) = I_Interpolated;
I1LRU(start_pos(1):ratio:end,start_pos(2):ratio:end,:) = I_Interpolated;

for ii = 1 : b
    t = I1LRU(:,:,ii); 
    t = imfilter(t',BaseCoeff,'circular'); 
    I1LRU(:,:,ii) = imfilter(t',BaseCoeff,'circular'); 
end

I_Interpolated = I1LRU;

end