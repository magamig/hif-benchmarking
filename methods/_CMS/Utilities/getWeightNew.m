function [Wi,index] = getWeightNew(Y3d, par)
% Y3d: �����ͼ�񣬱�����ά��ʽ

uGrpPatchs = Im2Patch3D(Y3d,par);                   
YClusters = ReshapeTo2D_C(uGrpPatchs);
n = par.nCluster;
label = kmeansPlus(YClusters,n);
Groups = cell(1,n);
index = cell(1,n);
Wi = cell(1,n);

for i = 1:n
    index{i}=find(label==i);
    Groups{i} = uGrpPatchs(:,:,index{i});
    Yi = Groups{i};
    Yi2d = ReshapeTo2D_C(Yi);
%     Wi{i} = LSR2(Yi2d,par.lambda);
    Wi{i} = LSR1(Yi2d,par.lambda);
end
