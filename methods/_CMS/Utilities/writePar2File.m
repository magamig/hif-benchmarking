function writePar2File(fp,par)

fprintf(fp,'Parameter of FBP and lock matching :\n\t');
fprintf(fp,'par.patsize: %3d \n\t',par.patsize);
fprintf(fp,'par.Pstep: %3d \n\t',par.Pstep);
fprintf(fp,'par.nCluster: %3d \n\n',par.nCluster);

% fprintf(fp,'par.patnum: %3d \n\t',par.patnum);
% fprintf(fp,'par.step: %3d \n\n',par.step);

fprintf(fp,'Parameter of algorithm 1 :\n\t');
fprintf(fp,'par.iter: %3d \n\t',par.iter);
fprintf(fp,'par.mu: %5.5f \n\t',par.mu);
fprintf(fp,'par.eta: %5.5f \n\t',par.eta);
fprintf(fp,'par.rho: %5.5f \n\t',par.rho);
fprintf(fp,'par.lambda: %5.5f \n\n',par.lambda);

