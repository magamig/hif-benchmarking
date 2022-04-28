
Improving Dictionary Learning README
November 28, 2012

Demo  example is in demoImproveD.m

% =========================================================================
%          Demo of Multiple Dictionary Updates and Coefficient Reuse
% =========================================================================
%   This file folder reproduces the Figures for paper:
%   "Improving Dictionary Learning: Multiple Dictionary Updates and Coefficient Reuse"
%   by Leslie N. Smith and Michael Elad, IEEE Signal Processing Letters, 
%	Vol 20, Issue 1, DOI.10.1109/LSP.2012.2229976 (2013)
%
%   Figures produced here will vary somewhat from those in the paper
%   because of the randomness in the method (i.e., initializing the
%   dictionary, dictionary training, etc.)
% =========================================================================


Installation:
--------------

1. Unpack the contents of the compressed file to a new directory, named e.g. "ImproveD".
2. The folder "private" contains  mex files that should be compiled for your system.
   If you have not done so before, configure Matlab's MEX compiler by entering
    >> mex -setup
   prior to using "make.m" in this folder For optimal performance, it is recommended that you select a 
   compiler that performs optimizations. For instance, in Windows, MS Visual Studio is preferred to Lcc.


Comments and questions can be emailed to Leslie N. Smith (leslie.smith@nrl.navy.mil) or Michael Elad (elad@cs.technion.ac.il).
