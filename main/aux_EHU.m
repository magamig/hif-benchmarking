load("data/GT/EHU/Pavia.mat");
hcube = hypercube(pavia,((1:size(pavia,3))*4)+420);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/Pavia.mat", "msi")

load("data/GT/EHU/PaviaU.mat");
hcube = hypercube(paviaU,((1:size(paviaU,3))*4)+420);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/PaviaU.mat", "msi")

load("data/GT/EHU/Indian_pines.mat");
hcube = hypercube(indian_pines,[1:103,109:149,164:219,221:240]*10+400);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/Indian_pines.mat", "msi")

load("data/GT/EHU/Salinas.mat");
hcube = hypercube(salinas,[1:107,113:153,168:223,225:244]*10+400);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/Salinas.mat", "msi")

load("data/GT/EHU/SalinasA.mat");
hcube = hypercube(salinasA,[1:107,113:153,168:223,225:244]*10+400);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/SalinasA.mat", "msi")

load("data/GT/EHU/Cuprite.mat");
hcube = hypercube(X,[1:107,113:153,168:223,225:244]*10+400);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/Cuprite.mat", "msi")

load("data/GT/EHU/KSC.mat");
hcube = hypercube(KSC,[1:103,113:149,168:203]*10+400);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/KSC.mat", "msi")

load("data/GT/EHU/Botswana.mat");
hcube = hypercube(Botswana,[0:45,72:87,92:109,124:154,177:210]*10+400);
msi = colorize(hcube,"Method","rgb","ContrastStretching",true);
save("data/MS/EHU/Botswana.mat", "msi")