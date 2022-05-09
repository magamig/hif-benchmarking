function P_rgb = load_nikon_rgb(startLambda,stepLambda,numLambda)

P_rgb = zeros(numLambda,3);

load('Nikon.mat');

for j = 1:3,    
    P_rgb(:,j) = interp1( wavelengths, Nikon_responses(:,j), startLambda:stepLambda:startLambda+stepLambda*(numLambda-1));
end

