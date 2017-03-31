function [Z] = zca2(x)

epsilon = 1e-4; % regularization parameter

% do not do dimensionality reduction
% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.


% PCA
sigma = x*x' / size(x,2);
[U, S, V] = svd(sigma);

Z = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;


