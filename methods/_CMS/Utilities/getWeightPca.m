function [Wi,index] = getWeightPca(Y3d, par)
% Y3d: 输入的图像，必须三维形式

uGrpPatchs = Im2Patch3D(Y3d,par);                   
YClusters = ReshapeTo2D(uGrpPatchs);
[coeff,score,latent] = pca(YClusters);
SelectNum = cumsum(latent)./sum(latent);
forwardNum = find(SelectNum>=0.99);
U_pca = score(:,1:forwardNum);
n = par.nCluster;
label = kmeansPlus(U_pca',n);
Groups = cell(1,n);
index = cell(1,n);
Wi = cell(1,n);

for i = 1:n
    index{i}=find(label==i);
    Groups{i} = uGrpPatchs(:,:,index{i});
    Yi = Groups{i};
    Yi2d = ReshapeTo2D_C(Yi);
    Wi{i} = LSR2(Yi2d,par.lambda);
end
