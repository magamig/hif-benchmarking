% hsi_path="data/HS/CAVE/4/balloons_ms.mat";msi_path="data/MS/CAVE/balloons_ms.mat";sri_path="data/SR/FUSE/CAVE/4/balloons_ms.mat";FUSE_run;

hsi = im2double(load(hsi_path).hsi);
msi = im2double(load(msi_path).msi);

sri = FUSE_wrapper(hsi,msi);
sri = im2uint16(sri);

save(sri_path, 'sri');