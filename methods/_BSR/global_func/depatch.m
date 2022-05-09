function [y_hat,final_numestimate]=depatch(Py_hat,patsize,y)
blockx = patsize;blocky = patsize;

final_numestimate = zeros(size(y));
final_extentestimate = zeros(size(y));
for indexi = 1:blocky
    for indexj = 1:blockx
        tempesti = reshape(Py_hat((indexi-1)*blockx+indexj,:),size(y)-[blockx,blocky]+1);
        numestimate = zeros(size(y));
        extentestimate = zeros(size(y));
        extentestimate(1:size(tempesti,1),1:size(tempesti,2)) = tempesti;
        numestimate(1:size(tempesti,1),1:size(tempesti,2)) = 1;
        
        extentestimate = circshift(extentestimate, [indexj,indexi]-1);
        numestimate = circshift(numestimate, [indexj,indexi]-1);
            
        final_numestimate = final_numestimate+numestimate;
        final_extentestimate = final_extentestimate+extentestimate;
    end
end
y_hat = final_extentestimate./final_numestimate;