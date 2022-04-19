% =========================================================================
% NSSR for Hyperspectral image super-resolution, Version 1.0
% Copyright(c) 2016 Weisheng Dong
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for Hyperspectral image super-
% resolution from a pair of low-resolution hyperspectral image and a high-
% resolution RGB image.
% 
% Please cite the following paper if you use this code:
%
% Weisheng Dong, Fazuo Fu, et. al.,"Hyperspectral image super-resolution via 
% non-negative structured sparse representation", IEEE Trans. On Image Processing, 
% vol. 25, no. 5, pp. 2337-2352, May 2015.
% 
%--------------------------------------------------------------------------

function  A   =  Nonnegative_SC( D, X, par )

A        =    zeros( size(D,2), size(X, 2) );
V        =    zeros( size(D,2), size(X, 2) );
T        =    100;
fun      =    zeros(T,1);
DTD       =   D'*D;
DTX       =   D'*X;
Ek        =   eye(par.K);
mu        =   0.005;
ro        =   1.07;
for  i  =  1:T
    S         =   inv(DTD + mu*Ek)*(DTX + mu*(A-V/(2*mu)) );
    A         =   max( soft(S+V/(2*mu), par.lambda/(2*mu)), 0);
    V         =   V + mu*( S - A );
    mu        =   mu*ro;

    fun(i)    =   0.5*sum(sum((X-D*A).^2)) + par.lambda*sum(sum(abs(A)));
end
