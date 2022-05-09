function [img] = mergePatch(p, y, x, Y, X)
img = zeros(Y, X);
coeff = zeros(Y, X);
p_idx = 1;
for xx=1:x
    for yy=1:y
        pp = col2im(p(p_idx,:), [y x], [Y X], 'sliding');
        img(yy:yy+Y-y,xx:xx+X-x) = img(yy:yy+Y-y,xx:xx+X-x)+pp;
        coeff(yy:yy+Y-y,xx:xx+X-x) = coeff(yy:yy+Y-y,xx:xx+X-x)+1;
        p_idx = p_idx+1;
    end
end
img = img ./ coeff;
return;

%     blockx = patsize;blocky = patsize;
%     final_numestimate = zeros(size(X(:,:,i)));
%     final_extentestimate = zeros(size(X(:,:,i)));
%     for indexi = 1:blocky
%         for indexj = 1:blockx
%             tempesti = reshape(Px_hat((indexi-1)*blockx+indexj,:),size(X(:,:,i))-[blockx,blocky]+1);
%             numestimate = zeros(size(X(:,:,i)));
%             extentestimate = zeros(size(X(:,:,i)));
%             extentestimate(1:size(tempesti,1),1:size(tempesti,2)) = tempesti;
%             numestimate(1:size(tempesti,1),1:size(tempesti,2)) = 1;          
%             
%             extentestimate = circshift(extentestimate, [indexj,indexi]-1);
%             numestimate = circshift(numestimate, [indexj,indexi]-1);
%             
%             final_numestimate = final_numestimate+numestimate;
%             final_extentestimate = final_extentestimate+extentestimate;
%         end
%     end
%     Xhat(:,:,i) = final_extentestimate./final_numestimate;

%     Xhat(:,:,i)=col2im(Px_hat,[patsize patsize],[size(X,1) size(X,2)]-1+patsize,'sliding');