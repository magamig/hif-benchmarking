function displayDic(D)

 blck_size = sqrt(size(D,1)); 
 no_blocks = size(D,2);
 no_rows = floor(sqrt(no_blocks));
 no_cols = ceil(no_blocks/no_rows);
 
 padd_size = 2;
 allreal = 0;
 
 color_dic = zeros((blck_size+padd_size)*no_rows, (blck_size+padd_size)*no_cols,3);
 
 Dreal = real(D);
 Dimag = imag(D);
 DD = [Dreal Dimag];
 
 if Dimag==0
     allreal = 1;
 end
 
 p=4.5;
M=max((DD(:)));
m=min((DD(:)));
if (m >= 0)
    me=0;
    sig=sqrt(mean(((DD(:))).^2));
else
    me=mean(DD(:));
    sig=sqrt(mean(((DD(:)-me)).^2));
end
DD=DD-me;
DD=min(max(DD,-p*sig),p*sig);
M=max((DD(:)));
m=min((DD(:)));
DD=(DD-m)/(M-m);

Dreal = DD(:,1:no_blocks);
Dimag = DD(:,no_blocks+1:2*no_blocks);



 for i=1:no_cols
     for j=1:no_rows
         if ((i-1)*no_rows+j) <= no_blocks
             color_dic(1+(blck_size+padd_size)*(j-1):(blck_size*j+padd_size*(j-1)),...
                 1+(blck_size+padd_size)*(i-1):(blck_size*i+padd_size*(i-1)),1) = ...,
                 reshape(Dreal(:,(i-1)*no_rows+j), blck_size,blck_size);
             color_dic(1+(blck_size+padd_size)*(j-1):(blck_size*j+padd_size*(j-1)),...
                 1+(blck_size+padd_size)*(i-1):(blck_size*i+padd_size*(i-1)),2) = ...,
                 reshape(Dimag(:,(i-1)*no_rows+j), blck_size,blck_size);
             color_dic(1+(blck_size+padd_size)*(j-1):(blck_size*j+padd_size*(j-1)),...
                 1+(blck_size+padd_size)*(i-1):(blck_size*i+padd_size*(i-1)),3) = ...,
                 0.5;
         end
     end
 end
 
  if allreal==1
     color_dic(:,:,2) = color_dic(:,:,1);
     color_dic(:,:,3) = color_dic(:,:,1);
 end
 

% color_dic(:,:,1) = real(all_dics);
% color_dic(:,:,2) = imag(all_dics);

imagesc((color_dic));
axis off;

% a = min(color_dic(:));
% b = max(color_dic(:));
% 
% figure(12);
% imagesc((color_dic-a)/(b-a)); colormap gray; axis off; 
% title(strcat('dictionaries'), ...
%      'FontName', 'TimesNewRoman','FontSize', 16)
 