% Compile and check all dependencies

% CAUTION! By running this script you agree to download the files from the
% internet
% ALTERNATIVELY you can manually download them by using the instructions
% in ./sisal/DOWNLOAD


% Compile Simplex Projection
cd reproject_simplex
ext = mexext;
if exist(['reproject_simplex_mex_fast.' ext],'file')~=3
    % mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" reproject_simplex_mex_fast.cpp
    mex reproject_simplex_mex_fast.cpp
end
cd ..

% Check if SISAL/SUNSAL package exists
if exist('./sisal/sunsal.m','file')~=2
    websave('./sisal/VCA.m','https://raw.githubusercontent.com/spidermanJie/hyperspectral_unmixing_project/master/unmixing_overview/include/VCA.m');
    websave('./sisal/hinge.m','https://raw.githubusercontent.com/spidermanJie/hyperspectral_unmixing_project/master/unmixing_overview/include/hinge.m');
    websave('./sisal/sisal.m','https://raw.githubusercontent.com/spidermanJie/hyperspectral_unmixing_project/master/unmixing_overview/include/sisal.m');
    websave('./sisal/soft.m','https://raw.githubusercontent.com/spidermanJie/hyperspectral_unmixing_project/master/unmixing_overview/include/soft.m');
    websave('./sisal/soft_neg.m','https://raw.githubusercontent.com/spidermanJie/hyperspectral_unmixing_project/master/unmixing_overview/include/soft_neg.m');
    websave('./sisal/sunsal.m','https://raw.githubusercontent.com/spidermanJie/hyperspectral_unmixing_project/master/unmixing_overview/include/sunsal.m');
end
