function [G] = create_G(roi, dsf)
% Create the Spatial Transform (uniform downsampling) Matrix G
w = dsf*(roi(2) - roi(1) + 1);%this is the pixel width of the full-resolution image (or image ROI)
%this width need not be equal to the image (or image ROI) height
%<->rectangles are ok
h = dsf*(roi(4) - roi(3) + 1);
Lc = w*h;
    
nzo = 1;
ci = zeros(1,Lc);
cj = zeros(1,Lc);
    
for col = 1:Lc/(dsf*dsf)
    fos = dsf*w*fix((col-1)/(w/dsf));
    sos = dsf*rem(col-1,(w/dsf));        
    cps = 1 + fos + sos;
        
    for jj = 0:(dsf*dsf - 1)
        cj(nzo+jj) = col;
    end
        
    for it = 1:dsf
        for ii = 0:(dsf-1)
            ci(nzo+ii) = cps+ii;
        end            
        nzo = nzo + dsf;
        cps = cps + w;
    end   
end
    
G = sparse(ci, cj, 1/(dsf*dsf), Lc, Lc/(dsf*dsf));
end

