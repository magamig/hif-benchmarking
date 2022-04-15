
###################################################################
#                                                                 #
#  Hyperspectral Super-Resolution by Coupled Spectral Unmixing    #
#                                                                 #
#    Charis Lanaras, Emmanuel Baltsavias and Konrad Schindler     #
#                          ICCV 2015                              #
#            Copyright 2015 ETH Zurich (Charis Lanaras)           #
#                                                                 #
###################################################################



ABOUT:
------
This is the authors' implementation of [1].

The code is implemented in MATLAB:
  compile.m                 - a script to check all dependencies
  demoSupResPALM.m          - a demo script that executes the algorithm
  SupResPALM.m              - the core part of the algorithm
  pompoms_ms.mat            - a sample image of the CAVE database (in 2D format)
  ./include/                - various dependecies (hyperspectal toolbox)
  ./reproject_simplex/      - Projection onto a Simplex (T. Pock)
  ./sisal/                  - SISAL/SUnSAL folder, files must be downloaded
  LICENSE                   - MIT license of the code
  README                    - this file



IMPORTANT:
----------
If you use this software you should cite the following in any resulting
publication:

    [1] Hyperspectral Super-Resolution by Coupled Spectral Unmixing
        C. Lanaras, E. Baltsavias, K. Schindler
        In ICCV, Santiago, Chile, December 2015



INSTALLING & RUNNING:
---------------------
Start MATLAB and run compile.m to build the utilities binaries.
This step can be omitted if you are using Windows 64 bit or Unix 64 bit,
since the binaries already exist.
However, you still need the SISAL/SUnSAL package.
	


NOTES:
------
1.  The simplex projection code (reproject_simplex_mex_fast.cpp) is courtesy
    of Thomas Pock, based on the on work of Yunmei Chen and Xiaojing Ye.
    Projection Onto A Simplex
    http://arxiv.org/abs/1101.6081

2.  The hyperspectral image (pompoms_ms.mat) is taken from the CAVE database.
    ALL COPYRIGHTS REMAIN WITH THE AUTHORS:
    F. Yasuma, T. Mitsunaga, D. Iso, and S.K. Nayar,    
    Generalized Assorted Pixel Camera: Post-Capture Control of Resolution, 
    Dynamic Range and Spectrum
    Technical Report, Department of Computer Science,
    Columbia University CUCS-061-08, Nov, 2008.

