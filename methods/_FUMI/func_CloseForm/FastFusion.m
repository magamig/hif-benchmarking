function [VXF] = FastFusion(MU,Q,Cc,InvDI,InvLbd,C3Cons,B2Sum,FBs,FBCNs,sizeMAT,sizeIM,n_dr,n_dc,dsf)
%% This function is to implement the fast fusion based on solving Sylvester Equation
%% Input: 
%MU:        The prior mean in frequence domain
%Q:         The invertible matrix decomposed from C1
%Cc:        EThe Cc in Eq. (27)
%InvDI:     In line 7, the inversion after C3
%InvLbd:    The inversion of Lambda matrix
%C3Cons:    The Cs in Eq. (27)
%B2Sum:     Auxillary matrix 1/d \sum_{i=1}^d \underline{D}_i
%FBs:       FFT of blurring kernel (matrix D in Step 1 of Algo.1)
%FBCNs:     The inversion of FBs (matrix D^{-1} in step 12 of Algo.1)
%sizeMAT:   [L nr*nc] 
%sizeIM:    [nr nc L] 
%n_dr,n_dc: The row and column of HS image
%dsf:       the convolution kernel
%% Output:
%VXF:       The FFT of fused image (in matrix form)
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
C3bar = C3Cons+PPlus(reshape((Cc*MU)',sizeIM).*FBs,n_dr,n_dc); % The implementation in frequency domain
MXsub = (C3bar-repmat(C3bar(1:n_dr,1:n_dc,:).*InvDI,[dsf dsf 1]).*B2Sum).*InvLbd;% InvDI= (1/d \sum D_i + \lambda_c^l In)^{-1}
% FBCNs represents the normalized conjugate of FBs: transform divide to multiply
VXF    = Q*reshape(PMinus(MXsub,n_dr,n_dc).*FBCNs,sizeMAT)';%% The invertible transforming %./FBs