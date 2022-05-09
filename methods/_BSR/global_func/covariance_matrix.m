 function c_xy = covariance_matrix(x,y)
% Calculate the covariance matrix of input images X and Y

[row_x,col_x,x_band] = size(x);
[row_y,col_y,y_band] = size(y);
if(row_x ~= row_y || col_x ~= col_y)
    disp('Spatial dimension mismatch between hyperpsectral and multispectral Image')
    return;
end

%Substract the mean of every band
% x=x-mean(x,3);y=y-mean(y,3);
for k = 1:x_band
    x(:,:,k)  = x(:,:,k) - mean2(x(:,:,k));
end
for k = 1:y_band
    y(:,:,k)  = y(:,:,k) - mean2(y(:,:,k));
end

c_xy = zeros(x_band, y_band);
for i=1:row_x 
    for j=1:col_x 
          cur_x = squeeze(x(i,j,:));
          cur_y = squeeze(y(i,j,:));
          c_xy = c_xy + cur_x*cur_y'; 
    end
end
%Make the estimation unbiased
c_xy = c_xy/(col_x*row_x-1);


