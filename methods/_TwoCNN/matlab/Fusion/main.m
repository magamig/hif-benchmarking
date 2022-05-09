clear all;
clc

addpath('D:\Code\Caffe-windows\caffe-windows-master\caffe-windows-master\matlab\demo');

load HSI;load MSI;


for num=1:size(HSI,3) % LR HSI
    HSI(:,:,num)=(HSI(:,:,num)-min(min(HSI(:,:,num))))/(max(max(HSI(:,:,num)))-min(min(HSI(:,:,num))));
end
for num=1:size(MSI,3) % HR HSI
    MSI(:,:,num)=(MSI(:,:,num)-min(min(MSI(:,:,num))))/(max(max(MSI(:,:,num)))-min(min(MSI(:,:,num))));
end




load X2_10000000.mat
X{1}=weights_conv11;X{2}=biases_conv11;
X{3}=weights_conv12;X{4}=biases_conv12;
X{5}=weights_conv13;X{6}=biases_conv13;
X{7}=weights_skip11;X{8}=biases_skip11;
X{9}=weights_conv21;X{10}=biases_conv21;
X{11}=weights_conv22;X{12}=biases_conv22;
X{13}=weights_conv23;X{14}=biases_conv23;
X{15}=weights_skip21;X{16}=biases_skip21;
X{17}=weights_fc11;X{18}=biases_fc11;
X{19}=weights_fc21;X{20}=biases_fc21;
X{21}=weights_fc1;X{22}=biases_fc1;
X{23}=weights_fc2;X{24}=biases_fc2;
X{25}=weights_fc3;X{26}=biases_fc3;



FI_HSI=zeros(256,256,162);

for row=16:size(HSI,1)-15

    for col=16:size(HSI,2)-15
        spe=HSI(row,col,1:162);
        spe=spe(:);

        spa=MSI(row-15:row+15,col-15:col+15,163:end);
        spe_FI=FI_CNN(X, spe, spa);
        FI_HSI(row-15,col-15,:)=spe_FI;
    end

end

% PSNRIn=zeros(1,162);
% PSNROut=zeros(1,162);
% for num=1:size(la_data,3)
%     PSNRIn(1,num) = 20*log10(1/sqrt(mean(mean((tr_data(:,:,num)-la_data(:,:,num)).^2))));
%     PSNROut(1,num) = 20*log10(1/sqrt(mean(mean((FI_HSI(:,:,num)-la_data(:,:,num)).^2))));
% end





