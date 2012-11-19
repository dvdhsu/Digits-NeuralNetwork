%% Much of this code was adapted from the code completed for the Coursera Machine Learning class. 
%% This neural network has two hidden layers, sized 410 and 100. Otherwise standard. 
%% Provided weights "weights.mat" are trained on the 60000 training set and receive 97.49% accuracy on MNIST test set. 


clear ; close all; clc

input_layer_size  = 784;  % 28 x 28  input image
hidden_layer1_size = 410; % 410 units in first hidden layer
hidden_layer2_size = 100; % 100 units in second hidden layer
num_labels = 10;          % 10 digits, or num_labels



fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

fprintf('\nTraining Neural Network... \n')

load('mnist_all.mat');

X = double([train1; train2; train3; train4; train5; train6; train7; train8; train9; train0]);
y = ones(60000, 1);

sizes = zeros(10, 1);

% Looks worse in a for-loop, so let's just do this:

sizes(1) = size(train1, 1);
sizes(2) = size(train2, 1);
sizes(3) = size(train3, 1);
sizes(4) = size(train4, 1);
sizes(5) = size(train5, 1);
sizes(6) = size(train6, 1);
sizes(7) = size(train7, 1);
sizes(8) = size(train8, 1);
sizes(9) = size(train9, 1);
sizes(10) = size(train0, 1);


pos = 1;
for i = 1:10
  y(pos:(pos + sizes(i) - 1)) = i;
  pos = pos + sizes(i);
end

options = optimset('MaxIter', 500);

% Regularization parameter:
lambda = 6;

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

testX = double([test1 ; test2 ; test3 ; test4 ; test5 ; test6 ; test7 ; test8 ; test9 ; test0]);
testY = ones(10000, 1);
sizes = zeros(10, 1);
sizes(1) = size(test1, 1);
sizes(2) = size(test2, 1);
sizes(3) = size(test3, 1);
sizes(4) = size(test4, 1);
sizes(5) = size(test5, 1);
sizes(6) = size(test6, 1);
sizes(7) = size(test7, 1);
sizes(8) = size(test8, 1);
sizes(9) = size(test9, 1);
sizes(10) = size(test0, 1);


pos = 1;
for i = 1:10
  testY(pos:(pos + sizes(i) - 1)) = i;
  pos = pos + sizes(i);
end


pred = predict(Theta1, Theta2, Theta3, testX);

fprintf('\nTest set accuracy: %f\n', mean(double(pred == testY) * 100));


