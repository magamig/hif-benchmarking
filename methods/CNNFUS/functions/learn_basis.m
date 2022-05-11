function [ E ] = learn_basis( Yh_up,p )
  vol = zeros(1, 20);
   max_vol = 0;
    for idx_VCA = 1:20
        E_aux = VCA(Yh_up,'Endmembers',p,'SNR',0,'verbose','off');
        vol(idx_VCA) = abs(det(E_aux'*E_aux));
        if vol(idx_VCA) > max_vol
            E = E_aux;
            max_vol = vol(idx_VCA);
        end   
    end


end

