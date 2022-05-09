
function  Y  =  Im2Patch3D( H, par)
% get full band patches
patsize     = par.patsize;
step = par.Pstep;

TotalPatNum = (floor((size(H,1)-patsize)/step)+1)*(floor((size(H,2)-patsize)/step)+1);                  %Total Patch Number in the image
Y           =   zeros(patsize*patsize, size(H,3), TotalPatNum);                                       %Patches in the original noisy image
k           =   0;

for i  = 1:patsize
    for j  = 1:patsize
        k     =  k+1;
        tempPatch     =  H(i:step:end-patsize+i,j:step:end-patsize+j,:);
        Y(k,:,:)      =  Unfold(tempPatch, size(tempPatch), 3);
    end
end         %Estimated Local Noise Level