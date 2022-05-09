
%% create RGB and false color images
contrastFactor = 2;

%% choose right name
assert(strcmp(name,'urban_park') || strcmp(name,'trees1_NE'), 'Unknown dataset');

switch name
    case 'urban_park'
        dataset = @(ch) ['urban_ch',num2str(ch),'_park'];    
    case 'trees1_NE'
        dataset = @(ch) ['trees1_ch',num2str(ch),'_NE'];
end 

if CreateColorImgs
    assert(strcmp(name,'urban_park'),...
    'RGB and NRG images only possible for urban data!')
    
    assert(length(channels)==4&&channels(1)==1&&channels(2)==2&&channels(3)==3&&channels(4)==4,...
        'Channels 1 to 4 are needed for the creation of color images')
end 


for lu = lambda_u
  for lk = lambda_k 
    maxi = -inf;
    imgs = cell(length(channels),1);
    ind = 0;
    for ch = channels;
      ind = ind+1;
      dataname = [dataset(ch)];
      imgname = [dataset(ch),'__PALM0',...
		'__lambda_u__',strrep(num2str(lu),'.','-'),...
		'__lambda_k_',strrep(num2str(lk),'.','-'),...
		'__gamma_0-9995__eta_0-003'];
      load(dataname)
      load(imgname)
      a = transform.a; 
      b = transform.b;
      u_resc = a * u + b;
      imgs{ind} = u_resc;
      max_data = a + b;
      
      if max_data>maxi
        maxi = max_data;
      end
    end
    
    ind = 0;
    for ch = channels
      ind = ind+1;
      imgname = [dataset(ch),'__PALM0',...
		'__lambda_u__',strrep(num2str(lu),'.','-'),...
		'__lambda_k_',strrep(num2str(lk),'.','-'),...
		'__gamma_0-9995__eta_0-003'];
      ext_imgname = [imgname,'.mat'];
      
      if maxi >1
        imgs{ind} = imgs{ind}./maxi;
      end 
      
      folder = which(ext_imgname);
      folder = folder(1:end-length(ext_imgname));
      C = 256;
      cd(folder)
      imwrite((C-1) * imgs{ind}, colormap(parula(C)),[imgname,'_resc','.png'])
      cd ../../../..
    end 
    
    if CreateColorImgs
       rgb = zeros([size(imgs{1}),3]);
       frgb = rgb;
       
       for i=1:3
           rgb(:,:,i)=imgs{i+1};
           frgb(:,:,i)=imgs{i};
       end
       
       
       cd(folder_results(12:end))
       rgb = contrastFactor*rgb;
       frgb = contrastFactor*frgb;
       imwrite(rgb,[name,'_rgb','_lambda_u_',strrep(num2str(lu),'.','-'),...
           '_lambda_k_',strrep(num2str(lk),'.','-'),'.png'])
       imwrite(frgb,[name,'_frgb','_lambda_u_',strrep(num2str(lu),'.','-'),...
           '_lambda_k_',strrep(num2str(lk),'.','-'),'.png'])      
    end
  end 	  
end
close all


