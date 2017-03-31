%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);   %50x81

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

L1 = W*x;                                  %50 x num images
L1Smooth = sqrt(L1.^2 + params.epsilon);  

%reconstruction
L2 = W'*L1 - x;                            %81 x num images

cost = (params.lambda*sum(L1Smooth(:))) + (0.5*sum(L2(:).^2));

% calculate gradient
grad_1 = (W * (2*L2) * x') + (2*L1*L2');     %50 x 81

% sparsity
grad_2 = (L1./L1Smooth) * x';

Wgrad = (0.5 * grad_1) + (params.lambda * grad_2); 

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);