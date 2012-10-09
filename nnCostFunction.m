function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

theta1_size = (input_layer_size + 1) * hidden_layer1_size;
theta2_size = (hidden_layer1_size + 1) * hidden_layer2_size;
theta3_size = (hidden_layer2_size + 1) * num_labels;

Theta1 = reshape(nn_params(1:theta1_size), hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((theta1_size + 1):(theta1_size + theta2_size)), hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((theta1_size + theta2_size + 1):end), num_labels, (hidden_layer2_size + 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%let's compute all the node values first
a1 = X'; % 400 x 5000, representing input
a1 = [ones(1, size(a1, 2)); a1]; % 401 x 5000, added the bias node

z2 = Theta1 * a1; % 100 x 5000
a2 = sigmoid(z2); % 100 x 5000
a2 = [ones(1, size(a2, 2)); a2]; % 101 x 5000, added the bias node

z3 = Theta2 * a2; % 25 x 5000
a3 = sigmoid(z3); % 25 x 5000

a3 = [ones(1, size(a3, 2)); a3]; % 26 x 5000, added the bias node

z4 = Theta3 * a3; % 10 x 5000
a4 = sigmoid(z4); % final output


% a strategy for generating logicalY taken from the forums. This is in contrast to creating a 
% zeros matrix then for-looping to set particular indicies to 1. 
identity = eye(num_labels);
logicalY = identity(:, y); % 10 x 5000

regularization = (lambda / (2 * m)) * (sum(sumsq(Theta1(:, 2:end))) + sum(sumsq(Theta2(:, 2:end))) + sum(sumsq(Theta2(:, 2:end))));

J = (1 / m) *   sum(sum((-logicalY .* log(a4)) - ((1 - logicalY) .* log(1 - a4)))) + regularization;

X = [ones(size(X, 1), 1), X];
Delta1 = Delta2 = Delta3 = 0;

delta4 = a4 - logicalY; % 10 x 5000, differences between predicted and actual, for each label
delta3 = (Theta3' * delta4) .* ([zeros(1, size(z3, 2)) ; sigmoidGradient(z3)]); % 26 x 5000
delta3 = delta3(2:end, :); % 25 x 5000

delta2 = (Theta2' * delta3) .* ([zeros(1, size(z2, 2)) ; sigmoidGradient(z2)]); % 101 x 5000
delta2 = delta2(2:end, :); % 100 x 5000

Delta1 = delta2 * a1';
Delta2 = delta3 * a2';	
Delta3 = delta4 * a3';

tempTheta1 = Theta1;
tempTheta2 = Theta2;
tempTheta3 = Theta3;

tempTheta1(:, 1) = zeros(size(Theta1, 1), 1);
tempTheta2(:, 1) = zeros(size(Theta2, 1), 1);
tempTheta3(:, 1) = zeros(size(Theta3, 1), 1);

delta1Regularization = (lambda / m) * tempTheta1;
delta2Regularizaiton = (lambda / m) * tempTheta2;
delta3Regularizaiton = (lambda / m) * tempTheta3;


Theta1_grad = Delta1 / m + delta1Regularization;
Theta2_grad = Delta2 / m + delta2Regularizaiton;
Theta3_grad = Delta3 / m + delta3Regularizaiton;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
