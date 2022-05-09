function [VXF] = FasterFusion(MU,Q,Cc,InvDI,InvLbd,C5Cons,FBs,FBCs,sizeMAT,sizeIM,n_dr,n_dc,dsf)
%% This function is to implement the fast fusion based on solving Sylvester Equation
%% Input: 
%MU:        The prior mean in frequence domain
%Q:         The invertible matrix decomposed from C1
%Cc:        EThe Cc in Eq. (27)
%InvDI:     In line 7, the inversion after C3
%InvLbd:    The inversion of Lambda matrix
%C5Cons:    The Cs in Eq. (27)
%FBs:       FFT of blurring kernel (matrix D in Step 1 of Algo.1)
%FBCs:      Conjugate of FBCs
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
%% Faster Fusion using matrix inverse lemma
% C5 = C5Cons+fft2(reshape((Cc*VXd_dec)',sizeIM)).*InvLbd;
C5bar = C5Cons+reshape((Cc*MU)',sizeIM).*InvLbd;
temp  = PPlus_s(C5bar/(dsf^2).*FBs,n_dr,n_dc); % The operation: temp=1/d*C5bar*Dv
invQUF = C5bar-repmat(temp.*InvDI,[dsf dsf 1]).*FBCs; % The operation: C5bar- temp*(\lambda_j d Im+\Sum_i=1^d Di^2)^{-1}Dv^H)
VXF    = Q*reshape(invQUF,sizeMAT)';