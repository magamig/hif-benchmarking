function showResults(RZ2d,AZ2d,sf)
% usage: type 'showResults(RZ2d,Z,sf)' in command line after run main

rzSize = [512,512,31];
sz = [rzSize(1),rzSize(2)];
P = create_P(); 
H = create_H(sz,sf);
X = RZ2d*H;

BZ2d = base_bicubic(X,sf); 
BY = P*BZ2d;
BY3d = ReshapeTo3D(BY,[rzSize(1),rzSize(2),3]);
imshow(BY3d),title('双三次插值后的RGB图')

RY = P*RZ2d;
RY3d = ReshapeTo3D(RY,[rzSize(1),rzSize(2),3]);
figure,imshow(RY3d),title('原始数据RGB图')

AY = P*AZ2d;
AY3d = ReshapeTo3D(AY,[rzSize(1),rzSize(2),3]);
figure,imshow(AY3d),title('重建后的RGB图')


