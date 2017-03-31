function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

for imageNum = 1:numImages
  for filterNum = 1:numFilters
    pooledImage = zeros(convolvedDim/poolDim, convolvedDim/poolDim);
    im = squeeze(convolvedFeatures(:, :, filterNum, imageNum));
    pooledImage = conv2(im, ones(poolDim, poolDim), "valid");
    pooledImage = pooledImage(1:poolDim:length(pooledImage), 1:poolDim:length(pooledImage));    
    pooledImage = pooledImage ./ (poolDim^2);
    pooledFeatures(:, :, filterNum, imageNum) = pooledImage;
  end
 end
 
 
 
end

