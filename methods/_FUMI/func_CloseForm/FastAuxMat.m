function [Q,Cc,InvDI,InvLbd,C3Cons] = FastAuxMat(yyh,yym,FBs,FBCs,sizeIM,n_dr,n_dc,InvCov,ER1E,RER2RE,B2Sum) 
%FastAuxMat(yyh,yym,FBs,FBCs,E,R,dsf,nr,nc,n_dr,n_dc,p,InvCov,R1,R2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input:
%yyh:       Normalized HS data
%yym:       Normalized MS data
%FBs:       FFT of blurring kernel
%FBCs:      Conjugate of FBs
%sizeIM:    [nr nc L], where nr and nc are the row and column of reference
%           image, L is the dimension of subspace
%n_dr,n_dc: The row and column of the HS image
%InvCov:    Covariance matrix of the Gaussian prior
%ER1E:      Auxillary matrix E*inv(Cov_HS)*E
%RER2RE:    Auxillary matrix (RE)*inv(Cov_HS)*(RE)
%B2Sum:     Auxillary matrix 1/d \sum_{i=1}^d \underline{D}_i (the first n_dr rows and first n_dc columns)
%% Output: 
%Q:         The invertible matrix decomposed from C1
%Cc:        The Cc in Eq. (27)
%InvDI:     In line 7, the inversion after C3
%InvLbd:    The inversion of Lambda matrix
%C3Cons:    The Cs in Eq. (27)

%If you use this code, please cite the following paper:

% [1] Qi Wei, Nicolas Dobigeon and Jean-Yves Tourneret, Fast Fusion of Multi-Band Images Based on Solving a Sylvester Equationn", submitted.
%% -------------------------------------------------------------------------
%
% Copyright (March, 2015):        Qi WEI (qi.wei@n7.fr)
%
% FastAuxMat is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------
[Q,Lambda]=eig(ER1E\(RER2RE+InvCov)); %% Eigendecomposition of C1 % C1=ER1E\(RER2RE+InvCov);
Lambda=reshape(diag(Lambda),[1 1 sizeIM(3)]);
Cc=(ER1E*Q)\InvCov; 
temp=(ER1E*Q)\eye(sizeIM(3)); % temp=Cc/InvCov; Temporay varaible
InvDI=1./(B2Sum(1:n_dr,1:n_dc,:)+repmat(Lambda,[n_dr n_dc 1]));
InvLbd=1./repmat(Lambda,[sizeIM(1) sizeIM(2) 1]);
C3Cons=PPlus((fft2(reshape((temp*yyh)',sizeIM)).*FBCs+fft2(reshape((temp*yym)',sizeIM))).*FBs,n_dr,n_dc);
% tic;C2Cons=PPlus(fft2(reshape((Cc/mu*(ConvC(yyh,FBCs(:,:,1),nr)+yym))',[nr nc p])).*FBs,n_dr,n_dc);toc %mask.*yyh