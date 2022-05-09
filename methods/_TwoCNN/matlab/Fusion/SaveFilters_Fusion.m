caffe.reset_all();
clear; close all;
%% settings
folder = 'D:\Code\Caffe-windows\caffe-windows-master\caffe-windows-master\matlab\fusion\';  % your directory of matlab in caffe
model = [folder 'super_resolution_train_test_7_mat - Copy.prototxt'];
weights = [folder 'super_resolution_iter_10000000.caffemodel'];  % put the caffemodel in the folder
savepath = [folder 'x2_10000000.mat'];
%layers = 9;

%% load model using mat_caffe
net = caffe.Net(model,weights,'test');

%% reshap parameters
% save conv11
conv_filters = net.layers(['conv11']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv11 = weights;

% save conv12
conv_filters = net.layers(['conv12']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv12 = weights;

% save bn_conv12
% mean12 = net.layers(['bn_conv12']).params(1).get_data();
% var12 = net.layers(['bn_conv12']).params(2).get_data();
% scale12 = net.layers(['bn_conv12']).params(3).get_data();

% save scale_conv12
% gamma12 = net.layers(['scale_conv12']).params(1).get_data();
% beta12 = net.layers(['scale_conv12']).params(2).get_data();

% save conv13
conv_filters = net.layers(['conv13']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv13 = weights;

% save bn_conv13
% mean13 = net.layers(['bn_conv13']).params(1).get_data();
% var13 = net.layers(['bn_conv13']).params(2).get_data();
% scale13 = net.layers(['bn_conv13']).params(3).get_data();

% save scale_conv13
% gamma13 = net.layers(['scale_conv13']).params(1).get_data();
% beta13 = net.layers(['scale_conv13']).params(2).get_data();


% save conv14
% conv_filters = net.layers(['conv14']).params(1).get_data();
% [fsize_w,fsize_h,channel,fnum] = size(conv_filters);
% if channel == 1
%     weights = single(ones(fsize_h*fsize_w, fnum));
% else
%     weights = single(ones(channel, fsize_h*fsize_w, fnum));
% end
% for i = 1 : channel
%     for j = 1 : fnum
%          temp = conv_filters(:,:,i,j);
%          if channel == 1
%             weights(:,j) = temp(:);
%          else
%             weights(i,:,j) = temp(:);
%          end
%     end
% end
% weights_conv14 = weights;

% save skip11
conv_filters = net.layers(['skip11']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_skip11 = weights;

% save bn_skip11
% mean_skip11 = net.layers(['bn_skip11']).params(1).get_data();
% var_skip11 = net.layers(['bn_skip11']).params(2).get_data();
% scale_skip11 = net.layers(['bn_skip11']).params(3).get_data();

% save scale_skip11
% gamma_skip11 = net.layers(['scale_skip11']).params(1).get_data();
% beta_skip11 = net.layers(['scale_skip11']).params(2).get_data();

% save skip12
% conv_filters = net.layers(['skip12']).params(1).get_data();
% [fsize_w,fsize_h,channel,fnum] = size(conv_filters);
% if channel == 1
%     weights = single(ones(fsize_h*fsize_w, fnum));
% else
%     weights = single(ones(channel, fsize_h*fsize_w, fnum));
% end
% for i = 1 : channel
%     for j = 1 : fnum
%          temp = conv_filters(:,:,i,j);
%          if channel == 1
%             weights(:,j) = temp(:);
%          else
%             weights(i,:,j) = temp(:);
%          end
%     end
% end
% weights_skip12 = weights;

% save conv21
conv_filters = net.layers(['conv21']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv21 = weights;

% save conv22
conv_filters = net.layers(['conv22']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv22 = weights;


% save bn_conv22
% mean22 = net.layers(['bn_conv22']).params(1).get_data();
% var22 = net.layers(['bn_conv22']).params(2).get_data();
% scale22 = net.layers(['bn_conv22']).params(3).get_data();

% save scale_conv22
% gamma22 = net.layers(['scale_conv22']).params(1).get_data();
% beta22 = net.layers(['scale_conv22']).params(2).get_data();


% save conv23
conv_filters = net.layers(['conv23']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv23 = weights;


% save bn_conv23
% mean23 = net.layers(['bn_conv23']).params(1).get_data();
% var23 = net.layers(['bn_conv23']).params(2).get_data();
% scale23 = net.layers(['bn_conv23']).params(3).get_data();

% save scale_conv23
% gamma23 = net.layers(['scale_conv23']).params(1).get_data();
% beta23 = net.layers(['scale_conv23']).params(2).get_data();

% save conv24
% conv_filters = net.layers(['conv24']).params(1).get_data();
% [fsize_w,fsize_h,channel,fnum] = size(conv_filters);
% if channel == 1
%     weights = single(ones(fsize_h*fsize_w, fnum));
% else
%     weights = single(ones(channel, fsize_h*fsize_w, fnum));
% end
% for i = 1 : channel
%     for j = 1 : fnum
%          temp = conv_filters(:,:,i,j);
%          if channel == 1
%             weights(:,j) = temp(:);
%          else
%             weights(i,:,j) = temp(:);
%          end
%     end
% end
% weights_conv24 = weights;

% save skip21
conv_filters = net.layers(['skip21']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_skip21 = weights;


% save bn_skip21
% mean_skip21 = net.layers(['bn_skip21']).params(1).get_data();
% var_skip21 = net.layers(['bn_skip21']).params(2).get_data();
% scale_skip21 = net.layers(['bn_skip21']).params(3).get_data();

% save scale_skip21
% gamma_skip21 = net.layers(['scale_skip21']).params(1).get_data();
% beta_skip21 = net.layers(['scale_skip21']).params(2).get_data();

% % save skip22
% conv_filters = net.layers(['skip22']).params(1).get_data();
% [fsize_w,fsize_h,channel,fnum] = size(conv_filters);
% if channel == 1
%     weights = single(ones(fsize_h*fsize_w, fnum));
% else
%     weights = single(ones(channel, fsize_h*fsize_w, fnum));
% end
% for i = 1 : channel
%     for j = 1 : fnum
%          temp = conv_filters(:,:,i,j);
%          if channel == 1
%             weights(:,j) = temp(:);
%          else
%             weights(i,:,j) = temp(:);
%          end
%     end
% end
% weights_skip22 = weights;

%%
% save ip11
fc_filters = net.layers(['ip11']).params(1).get_data();
weights_fc11=fc_filters;
% save bn_ip11
% mean_ip11 = net.layers(['bn_ip11']).params(1).get_data();
% var_ip11 = net.layers(['bn_ip11']).params(2).get_data();
% scale_ip11 = net.layers(['bn_ip11']).params(3).get_data();
% save scale_ip11
% gamma_ip11 = net.layers(['scale_ip11']).params(1).get_data();
% beta_ip11 = net.layers(['scale_ip11']).params(2).get_data();

% save ip21
fc_filters = net.layers(['ip21']).params(1).get_data();
weights_fc21=fc_filters;
% save bn_ip21
% mean_ip21 = net.layers(['bn_ip21']).params(1).get_data();
% var_ip21 = net.layers(['bn_ip21']).params(2).get_data();
% scale_ip21 = net.layers(['bn_ip21']).params(3).get_data();
% save scale_ip21
% gamma_ip21 = net.layers(['scale_ip21']).params(1).get_data();
% beta_ip21 = net.layers(['scale_ip21']).params(2).get_data();

% save ip1
fc_filters = net.layers(['ip1']).params(1).get_data();
weights_fc1=fc_filters;
% save bn_ip1
% mean_ip1 = net.layers(['bn_ip1']).params(1).get_data();
% var_ip1 = net.layers(['bn_ip1']).params(2).get_data();
% scale_ip1 = net.layers(['bn_ip1']).params(3).get_data();
% save scale_ip1
% gamma_ip1 = net.layers(['scale_ip1']).params(1).get_data();
% beta_ip1 = net.layers(['scale_ip1']).params(2).get_data();

% save ip2
fc_filters = net.layers(['ip2']).params(1).get_data();
weights_fc2=fc_filters;
% save bn_ip2
% mean_ip2 = net.layers(['bn_ip2']).params(1).get_data();
% var_ip2 = net.layers(['bn_ip2']).params(2).get_data();
% scale_ip2 = net.layers(['bn_ip2']).params(3).get_data();
% save scale_ip2
% gamma_ip2 = net.layers(['scale_ip2']).params(1).get_data();
% beta_ip2 = net.layers(['scale_ip2']).params(2).get_data();



% save ip3
fc_filters = net.layers(['ip3']).params(1).get_data();
weights_fc3=fc_filters;




%% save parameters
biases_conv11 = net.layers('conv11').params(2).get_data();
biases_conv12 = net.layers('conv12').params(2).get_data();
biases_conv13 = net.layers('conv13').params(2).get_data();
% biases_conv14 = net.layers('conv14').params(2).get_data();
biases_skip11 = net.layers('skip11').params(2).get_data();
% biases_skip12 = net.layers('skip12').params(2).get_data();
biases_conv21 = net.layers('conv21').params(2).get_data();
biases_conv22 = net.layers('conv22').params(2).get_data();
biases_conv23 = net.layers('conv23').params(2).get_data();
% biases_conv24 = net.layers('conv24').params(2).get_data();
biases_skip21 = net.layers('skip21').params(2).get_data();
% biases_skip22 = net.layers('skip22').params(2).get_data();
biases_fc11 = net.layers('ip11').params(2).get_data();
biases_fc21 = net.layers('ip21').params(2).get_data();
biases_fc1 = net.layers('ip1').params(2).get_data();
biases_fc2 = net.layers('ip2').params(2).get_data();
biases_fc3 = net.layers('ip3').params(2).get_data();



save(savepath,'weights_conv11','biases_conv11','weights_conv12','biases_conv12','weights_conv13','biases_conv13','weights_skip11','biases_skip11',...
    'weights_conv21','biases_conv21','weights_conv22','biases_conv22','weights_conv23','biases_conv23','weights_skip21','biases_skip21',...
    'weights_fc11','biases_fc11','weights_fc21','biases_fc21','weights_fc1','biases_fc1','weights_fc2','biases_fc2','weights_fc3','biases_fc3');

