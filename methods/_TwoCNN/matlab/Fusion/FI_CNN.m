function spe_h = FI_CNN(X, spe, spa)

%% load CNN model parameters
weights_conv11=X{1};biases_conv11=X{2};
weights_conv12=X{3};biases_conv12=X{4};
weights_conv13=X{5};biases_conv13=X{6};
weights_skip11=X{7};biases_skip11=X{8};
weights_conv21=X{9};biases_conv21=X{10};
weights_conv22=X{11};biases_conv22=X{12};
weights_conv23=X{13};biases_conv23=X{14};
weights_skip21=X{15};biases_skip21=X{16};
weights_fc11=X{17};biases_fc11=X{18};
weights_fc21=X{19};biases_fc21=X{20};
weights_fc1=X{21};biases_fc1=X{22};
weights_fc2=X{23};biases_fc2=X{24};
weights_fc3=X{25};biases_fc3=X{26};

keyboard

[hei_spe, wid_spe] = size(spe);
[hei_spa, wid_spa] = size(spa);

%% conv11
weights_conv11 = reshape(weights_conv11, size(weights_conv11,1), 1, size(weights_conv11,2));
conv11_data = zeros(hei_spe-size(weights_conv11,1)+1, 1, size(weights_conv11,3));
for i = 1 : size(weights_conv11,3)
    conv11_data(:,:,i) = filter2(weights_conv11(:,:,i), spe, 'valid');
    conv11_data(:,:,i) = max(conv11_data(:,:,i) + biases_conv11(i), 0);
end

%% pool11
% conv11_pool=zeros(ceil((size(conv11_data,1)-2)/2)+1,1,size(conv11_data,3));
% for i = 1 : size(weights_conv11,3)
%     conv11_pool(:,:,i) = Pooling_my(2, 1, 2, conv11_data(:,:,i));
% end

%% conv12
conv12_data = zeros(size(conv11_data,1)-size(weights_conv12,2)+1, 1, size(weights_conv12,3));
for i = 1 : size(weights_conv12,3)
    for j = 1 : size(weights_conv11,3)
        filt_conv12 = reshape(weights_conv12(j,:,i),45,1);
        conv12_data(:,:,i) = conv12_data(:,:,i) + filter2(filt_conv12, conv11_data(:,:,j), 'valid');
    end
    conv12_data(:,:,i) = max(conv12_data(:,:,i) + biases_conv12(i), 0);
end

%% bn_conv12
% for i=1 : size(conv12_data,3)
%     mean = mean12(i)/scale12;
%     var = var12(i)/scale12;
%     conv12_data(:,:,i) = (conv12_data(:,:,i) - mean)/sqrt(var);
%     conv12_data(:,:,i) = max(gamma12(i)*conv12_data(:,:,i) + beta12(i), 0);
% end

%% pool12
% conv12_pool=zeros(size(conv12_data,1)/2,1,size(conv12_data,3));
% for i = 1 : size(weights_conv12,3)
%     conv12_pool(:,:,i) = Pooling_my(2, 1, 2, conv12_data(:,:,i));
% end

%% skip11
skip11_data = zeros(size(conv11_data,1)-size(weights_skip11,2)+1, 1, size(weights_skip11,3));
for i = 1 : size(weights_skip11,3)
    for j = 1 : size(weights_conv11,3)
        filt_skip11 = reshape(weights_skip11(j,:,i),89,1);
        skip11_data(:,:,i) = skip11_data(:,:,i) + filter2(filt_skip11, conv11_data(:,:,j), 'valid');
    end
    skip11_data(:,:,i) = max(skip11_data(:,:,i) + biases_skip11(i), 0);
end

%% bn_skip11
% for i=1 : size(skip11_data,3)
%     mean = mean_skip11(i)/scale_skip11;
%     var = var_skip11(i)/scale_skip11;
%     skip11_data(:,:,i) = (skip11_data(:,:,i) - mean)/sqrt(var);
%     skip11_data(:,:,i) = max(gamma_skip11(i)*skip11_data(:,:,i) + beta_skip11(i), 0);
% end

%% conv13
conv13_data = zeros(size(conv12_data,1)-size(weights_conv13,2)+1, 1, size(weights_conv13,3));
for i = 1 : size(weights_conv13,3)
    for j = 1 : size(conv12_data,3)
        filt_conv13 = reshape(weights_conv13(j,:,i),45,1);
        conv13_data(:,:,i) = conv13_data(:,:,i) + filter2(filt_conv13, conv12_data(:,:,j), 'valid');
    end
    conv13_data(:,:,i) = max(conv13_data(:,:,i) + biases_conv13(i), 0);
end

%% bn_conv13
% for i=1 : size(conv13_data,3)
%     mean = mean13(i)/scale13;
%     var = var13(i)/scale13;
%     conv13_data(:,:,i) = (conv13_data(:,:,i) - mean)/sqrt(var);
%     conv13_data(:,:,i) = max(gamma13(i)*conv13_data(:,:,i) + beta13(i), 0);
% end

%% skip11+conv13
res11 = zeros(size(conv13_data));
for  i = 1 : size(conv13_data,3)
    res11(:,:,i) = max(skip11_data(:,:,i)+conv13_data(:,:,i), 0);
end

%% pool13
% conv13_pool=zeros(size(conv13_data,1)/2,1,size(conv13_data,3));
% for i = 1 : size(weights_conv13,3)
%     conv13_pool(:,:,i) = Pooling_my(2, 1, 2, conv13_data(:,:,i));
% end

%% conv14
% conv14_data = zeros(size(conv13_data,1)-size(weights_conv14,2)+1, 1, size(weights_conv14,3));
% for i = 1 : size(weights_conv14,3)
%     for j = 1 : size(weights_conv13,3)
%         filt_conv14 = reshape(weights_conv14(j,:,i),45,1);
%         conv14_data(:,:,i) = conv14_data(:,:,i) + filter2(filt_conv14, conv13_data(:,:,j), 'valid');
%     end
%     conv14_data(:,:,i) = max(conv14_data(:,:,i) + biases_conv14(i), 0);
% end

% res11 = zeros(size(conv14_data));
% for  i = 1 : size(conv14_data,3)
%     res11(:,:,i) = max(skip11_data(:,:,i)+conv14_data(:,:,i), 0);
% end


%% skip12
% skip12_data = zeros(size(res11,1)-size(weights_skip12,2)+1, 1, size(weights_skip12,3));
% for i = 1 : size(weights_skip12,3)
%     for j = 1 : size(res11,3)
%         filt_skip12 = reshape(weights_skip12(j,:,i),69,1);
%         skip12_data(:,:,i) = skip12_data(:,:,i) + filter2(filt_skip12, res11(:,:,j), 'valid');
%     end
%     skip12_data(:,:,i) = max(skip12_data(:,:,i) + biases_skip12(i), 0);
% end

%% conv15
% conv15_data = zeros(size(conv14_data,1)-size(weights_conv15,2)+1, 1, size(weights_conv15,3));
% for i = 1 : size(weights_conv15,3)
%     for j = 1 : size(weights_conv14,3)
%         filt_conv15 = reshape(weights_conv15(j,:,i),30,1);
%         conv15_data(:,:,i) = conv15_data(:,:,i) + filter2(filt_conv15, conv14_data(:,:,j), 'valid');
%     end
%     conv15_data(:,:,i) = max(conv15_data(:,:,i) + biases_conv15(i), 0);
% end

%% skip12+conv14
% res12 = zeros(size(conv14_data));
% for  i = 1 : size(conv14_data,3)
%     res12(:,:,i) = max(skip12_data(:,:,i)+conv14_data(:,:,i), 0);
% end

%% flatten
fla_spe=reshape(res11,size(res11,1)*size(res11,2)*size(res11,3),1);

%% conv21
conv21_data = zeros(size(spa,1)-sqrt(size(weights_conv21,2))+1, size(spa,2)-sqrt(size(weights_conv21,2))+1, size(weights_conv21,3));
for i = 1 : size(weights_conv21,3)
    for j = 1 : size(spa,3)
        tmp = reshape(weights_conv21(j,:,i),10,10);
        filt_conv21 = tmp';
        conv21_data(:,:,i) = conv21_data(:,:,i) + filter2(filt_conv21, spa(:,:,j), 'valid');
    end
    conv21_data(:,:,i) = max(conv21_data(:,:,i) + biases_conv21(i), 0);
end
% weights_conv21 = reshape(weights_conv21, size(weights_conv21,1), 1, size(weights_conv21,2));
% conv21_data = zeros(hei_spa-sqrt(size(weights_conv21,1))+1, hei_spa-sqrt(size(weights_conv21,1))+1, size(weights_conv21,3));
% for i = 1 : size(weights_conv21,3)
%     tmp = reshape(weights_conv21(:,:,i),10,10);
%     filt_conv21 = tmp';
%     conv21_data(:,:,i) = filter2(filt_conv21, spa, 'valid');
%     conv21_data(:,:,i) = max(conv21_data(:,:,i) + biases_conv21(i), 0);
% end
%% pool21
% conv21_pool=single(zeros(floor(size(conv21_data,1)/2)+1,floor(size(conv21_data,2)/2)+1,size(conv21_data,3)));
% for i = 1 : size(weights_conv21,3)
%     conv21_pool(:,:,i) = Pooling_my(2, 2, 2, conv21_data(:,:,i));
% end


%% conv22
conv22_data = zeros(size(conv21_data,1)-sqrt(size(weights_conv22,2))+1, size(conv21_data,2)-sqrt(size(weights_conv22,2))+1, size(weights_conv22,3));
for i = 1 : size(weights_conv22,3)
    for j = 1 : size(conv21_data,3)
        tmp = reshape(weights_conv22(j,:,i),10,10);
        filt_conv22 = tmp';
        conv22_data(:,:,i) = conv22_data(:,:,i) + filter2(filt_conv22, conv21_data(:,:,j), 'valid');
    end
    conv22_data(:,:,i) = max(conv22_data(:,:,i) + biases_conv22(i), 0);
end

%% bn_conv22
% for i=1 : size(conv22_data,3)
%     mean = mean22(i)/scale22;
%     var = var22(i)/scale22;
%     conv22_data(:,:,i) = (conv22_data(:,:,i) - mean)/sqrt(var);
%     conv22_data(:,:,i) = max(gamma22(i)*conv22_data(:,:,i) + beta22(i), 0);
% end

%% skip21
skip21_data = zeros(size(conv21_data,1)-sqrt(size(weights_skip21,2))+1, size(conv21_data,2)-sqrt(size(weights_skip21,2))+1, size(weights_skip21,3));
for i = 1 : size(weights_skip21,3)
    for j = 1 : size(conv21_data,3)
        tmp = reshape(weights_skip21(j,:,i),19,19);
        filt_skip21 = tmp';
        skip21_data(:,:,i) = skip21_data(:,:,i) + filter2(filt_skip21, conv21_data(:,:,j), 'valid');
    end
    skip21_data(:,:,i) = max(skip21_data(:,:,i) + biases_skip21(i), 0);
end

%% bn_skip21
% for i=1 : size(skip21_data,3)
%     mean = mean_skip21(i)/scale_skip21;
%     var = var_skip21(i)/scale_skip21;
%     skip21_data(:,:,i) = (skip21_data(:,:,i) - mean)/sqrt(var);
%     skip21_data(:,:,i) = max(gamma_skip21(i)*skip21_data(:,:,i) + beta_skip21(i), 0);
% end

%% conv23
conv23_data = zeros(size(conv22_data,1)-sqrt(size(weights_conv23,2))+1, size(conv22_data,2)-sqrt(size(weights_conv23,2))+1, size(weights_conv23,3));
for i = 1 : size(weights_conv23,3)
    for j = 1 : size(conv22_data,3)
        tmp = reshape(weights_conv23(j,:,i),10,10);
        filt_conv23 = tmp';
        conv23_data(:,:,i) = conv23_data(:,:,i) + filter2(filt_conv23, conv22_data(:,:,j), 'valid');
    end
    conv23_data(:,:,i) = max(conv23_data(:,:,i) + biases_conv23(i), 0);
end

%% bn_conv23
% for i=1 : size(conv23_data,3)
%     mean = mean23(i)/scale23;
%     var = var23(i)/scale23;
%     conv23_data(:,:,i) = (conv23_data(:,:,i) - mean)/sqrt(var);
%     conv23_data(:,:,i) = max(gamma23(i)*conv23_data(:,:,i) + beta23(i), 0);
% end

%% skip21+conv23
res21 = zeros(size(conv23_data));
for  i = 1 : size(conv23_data,3)
    res21(:,:,i) = max(skip21_data(:,:,i)+conv23_data(:,:,i), 0);
end
%% conv24
% conv24_data = zeros(size(conv23_data,1)-sqrt(size(weights_conv24,2))+1, size(conv23_data,2)-sqrt(size(weights_conv24,2))+1, size(weights_conv24,3));
% for i = 1 : size(weights_conv24,3)
%     for j = 1 : size(conv23_data,3)
%         tmp = reshape(weights_conv24(j,:,i),8,8);
%         filt_conv24 = tmp';
%         conv24_data(:,:,i) = conv24_data(:,:,i) + filter2(filt_conv24, conv23_data(:,:,j), 'valid');
%     end
%     conv24_data(:,:,i) = max(conv24_data(:,:,i) + biases_conv24(i), 0);
% end

%% skip22
% skip22_data = zeros(size(res21,1)-sqrt(size(weights_skip22,2))+1, size(res21,2)-sqrt(size(weights_skip22,2))+1, size(weights_skip22,3));
% for i = 1 : size(weights_skip22,3)
%     for j = 1 : size(res21,3)
%         tmp = reshape(weights_skip22(j,:,i),15,15);
%         filt_conv23 = tmp';
%         skip22_data(:,:,i) = skip22_data(:,:,i) + filter2(filt_conv23, res21(:,:,j), 'valid');
%     end
%     skip22_data(:,:,i) = max(skip22_data(:,:,i) + biases_skip22(i), 0);
% end

%% conv25
% conv25_data = zeros(size(conv24_data,1)-sqrt(size(weights_conv25,2))+1, size(conv24_data,2)-sqrt(size(weights_conv25,2))+1, size(weights_conv25,3));
% for i = 1 : size(weights_conv25,3)
%     for j = 1 : size(conv24_data,3)
%         tmp = reshape(weights_conv25(j,:,i),6,6);
%         filt_conv25 = tmp';
%         conv25_data(:,:,i) = conv25_data(:,:,i) + filter2(filt_conv25, conv24_data(:,:,j), 'valid');
%     end
%     conv25_data(:,:,i) = max(conv25_data(:,:,i) + biases_conv25(i), 0);
% end

%% skip22+conv24
% res22 = zeros(size(conv24_data));
% for  i = 1 : size(conv24_data,3)
%     res22(:,:,i) = max(skip22_data(:,:,i)+conv24_data(:,:,i), 0);
% end

%% flatten
%fla_spa = reshape(conv23_data,size(conv23_data,1)*size(conv23_data,2)*size(conv23_data,3),1);
fla_spa = single(zeros(1,1));
for num=1:size(res21,3)
    tmp=res21(:,:,num);
    tmp=tmp';
    tmp=tmp(:);
    fla_spa=cat(1,fla_spa,tmp);
end
fla_spa(1,:)=[];

%% fc11
ip11 = max(weights_fc11'*fla_spe + biases_fc11, 0);
%% bn_fc11
% mean = mean_ip11./scale_ip11;
% var = var_ip11./scale_ip11;
% ip11 = (ip11 - mean)./sqrt(var);
% ip11 = gamma_ip11.*ip11 + beta_ip11;
% ip11 = max(ip11,0);

%% fc21
ip21 = max(weights_fc21'*fla_spa + biases_fc21, 0);
%% bn_fc21
% mean = mean_ip21./scale_ip21;
% var = var_ip21./scale_ip21;
% ip21 = (ip21 - mean)./sqrt(var);
% ip21 = gamma_ip21.*ip21 + beta_ip21;
% ip21 = max(ip21,0);

%% concatenate
fla = [ip11', ip21']';

%% fc1
ip1 = max(weights_fc1'*fla + biases_fc1, 0);
%% bn_fc1
% mean = mean_ip1./scale_ip1;
% var = var_ip1./scale_ip1;
% ip1 = (ip1 - mean)./sqrt(var);
% ip1 = gamma_ip1.*ip1 + beta_ip1;
% ip1 = max(ip1,0);

%% fc2
ip2 = max(weights_fc2'*ip1 + biases_fc2, 0);
%% bn_fc2
% mean = mean_ip2./scale_ip2;
% var = var_ip2./scale_ip2;
% ip2 = (ip2 - mean)./sqrt(var);
% ip2 = gamma_ip2.*ip2 + beta_ip2;
% ip2 = max(ip2,0);


ip2_fla = max(ip2+fla, 0);

%% fc3
ip3 = weights_fc3'*ip2_fla + biases_fc3;

spe_h=ip3;
















