function [dataHS_low_spatial, dataHS_truncated] = generate_low_spatial_res_measurement( dataHS );

% generate_low_spatial_res_measurement.m
dataWidth  = size(dataHS,1);
dataHeight = size(dataHS,2);
dataWavelengths = size(dataHS,3);

blockWidth  = 32;
blockHeight = 32; 
samplingKernel = ones(blockWidth,blockHeight);
samplingKernel = samplingKernel / sum(sum(samplingKernel));

numWBlocks = floor( size(dataHS,1) / blockWidth  );
numHBlocks = floor( size(dataHS,2) / blockHeight ); 

dataHS_truncated = dataHS( 1: blockWidth * numWBlocks, 1: blockHeight * numHBlocks, : );
dataHS_low_spatial = zeros(numWBlocks, numHBlocks, dataWavelengths);

for k = 1:dataWavelengths,
    for i = 1:numWBlocks,
        for j = 1:numHBlocks,
            dataHS_low_spatial(i,j,k) = sum(sum( samplingKernel .* dataHS_truncated(blockWidth*(i-1)+1:blockWidth*i,blockHeight*(j-1)+1:blockHeight*j,k) ));
        end
    end
end
