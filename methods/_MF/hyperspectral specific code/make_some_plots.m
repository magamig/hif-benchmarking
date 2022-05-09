% make_some_plots
%

[X_row_sort,ind] = sort(sum(abs(X),2),'descend');

figure(1);
subplot(2,1,1);
imagesc(A(:,ind));
colormap('jet');
subplot(2,1,2);
stem(X_row_sort);

[dontCare,coeffInds] = sort(abs(coefficients),3,'descend');

sigma = sign(coefficients);

for i = 1:512,
    for j = 1:512,
        sigma(i,j,:) = sigma(i,j,coeffInds(i,j,:));
    end
end

h = 2; 
figure(3); 
clf;
for i = 1:h, 
    for j = 1:3, 
        subplot(h,4,4*(i-1)+j); 
        imagesc(sum(sigma(:,:,1:j) .* (coeffInds(:,:,1:j) == ind(i)),3),[-1,1]); 
        axis off; 
    end; 
end; 
colormap('gray');

for i = 1:h, subplot(h,4,4*(i-1)+4); plot(A(:,ind(i)),'-','LineWidth',2); end;

errorHS = dataHS - reconstructed_HS;
figure(4);
imagesc(2^-8 * sqrt(sum(errorHS .* errorHS,3) / size(errorHS,3)));
colorbar;