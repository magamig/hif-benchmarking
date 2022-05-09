function [X] = create_X(Z, G, roi, dsf, rescale)
% Create the Low Resolution Hyperspectral X
X = rescale*(Z*G); % when G is known to us 

%% When G is unknown to us, we use resize Matlab function... 

% Mh = size(Z,1);
% ImgSize = roi*dsf;
% X = zeros(Mh,roi(2)*roi(4));
% for i = 1:Mh
%     Img = reshape(Z(i,:),ImgSize(2),ImgSize(4)); 
%     LX = imresize(Img, [roi(2) roi(4)], 'bicubic');
%     X(i,:) = LX(:)';
% end
% X = rescale*X;
end