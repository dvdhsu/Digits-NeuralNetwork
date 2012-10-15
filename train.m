%% Much of this code was adapted from the code completed for the Coursera Machine Learning class. 
%% This neural network has two hidden layers, sized 150 and 300. Otherwise standard. 
%% Provided weights "weights.mat" are trained on the 60000 training set and receive 97.1% accuracy on MNIST test set. 


clear ; close all; clc

input_layer_size  = 784;  % 28 x 28  input image
hidden_layer1_size = 400; % 400 units in first hidden layer
hidden_layer2_size = 50; %  50 units in second hidden layer
num_labels = 10;          % 10 digits, or num_labels



fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

fprintf('\nTraining Neural Network... \n')

load('mnist_all_processed.mat');

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

% Regularization parameter:
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

theta1_size = (input_layer_size + 1) * hidden_layer1_size;
theta2_size = (hidden_layer1_size + 1) * hidden_layer2_size;
theta3_size = (hidden_layer2_size + 1) * num_labels;

Theta1 = reshape(nn_params(1:theta1_size), hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((theta1_size + 1):(theta1_size + theta2_size)), hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((theta1_size + theta2_size + 1):end), num_labels, (hidden_layer2_size + 1));


%% Test set accuracy:

pred = predict(Theta1, Theta2, Theta3, testX);

fprintf('\nTest set accuracy: %f\n', mean(double(pred == testY) * 100));


