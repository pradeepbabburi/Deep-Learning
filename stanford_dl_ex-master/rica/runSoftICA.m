%% We will use minFunc for this exercise, but you can use your
% own optimizer of choice
clear; close all; clc;
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled
%%addpath(genpath('../common/minFunc_2012/')) % path to minfunc
%% These parameters should give you sane results. We recommend experimenting
% with these values after you have a working solution.
global params;
params.m = 10000; % num patches
params.patchWidth = 18; % width of a patch
params.n = params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 50; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity

% Load MNIST data set
data = loadMNISTImages('../imagesMNIST/train-images-idx3-ubyte');

%% Preprocessing
% Our strategy is as follows:
% 1) Sample random patches in the images
% 2) Apply standard ZCA transformation to the data
% 3) Normalize each patch to be between 0 and 1 with l2 normalization
figure('name','Raw images');
randsel = randi(params.m,200,1); % A random selection of samples for visualization
display_network(data(:,randsel));

% Step 1) Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
figure('name','image patches');
display_network(patches(:,randsel));

% Step 2) Apply ZCA
patches = zca2(patches);
figure('name','zca image patches');
display_network(patches(:,randsel));
fprintf('Program paused. Enter to begin training. Ctrl-C to exit.\n');
pause;
% Step 3) Normalize each patch. Each patch should be normalized as
% x / ||x||_2 where x is the vector representation of the patch
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);

%% Run the optimization
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 500;
options.useMex = 0;
%options.display = 'off';
options.outputFcn = @showBases;

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);

%% check gradient
%fprintf('Checking numerical gradient.\n');
%[cost, grad] = softICACost(randTheta, x(:,10), params);
%numgrad = computeNumericalGradient(@(p) softICACost(p, x, params), randTheta);
%disp([numgrad grad]);
%fprintf('Gradient check finished. Enter to begin optimization. Ctrl-C to exit.\n');
%pause;
% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options); % Use x or xw

% display result
W = reshape(opttheta, params.numFeatures, params.n);
display_network(W');