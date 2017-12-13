function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

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

%% calculate cost function

% l1 = input_layer_size;
% l2 = hidden_layer_size;
% l3 = num_labels;

% ff to calculate all hypothesis. store value for each sample in a row.
a1_all = [ ones(m, 1), X];
z2_all = (Theta1*a1_all')';
a2_all = [ ones(m, 1), sigmoid(z2_all)];
z3_all = (Theta2*a2_all')';
a3_all = sigmoid(z3_all);

% labelling. each verdict is a vector of type one-vs-all
all_labels = eye(num_labels);

% first term corresonding to non regularized cost
s = 0;
for cnt = 1:m
    for cnt1 = 1:num_labels
        temp = -all_labels(y(cnt),cnt1)*log(a3_all(cnt,cnt1)) - (1 - all_labels(y(cnt),cnt1))*log(1 - a3_all(cnt,cnt1));
        s = s + temp;
    end
end
J_ir = 1/m*s;
% second term corresponding to reguarized part
Theta1(:,1) = 0; % we do not regularize bias intercepter
Theta2(:,1) = 0;
J_rg = lambda/2/m*(sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));

J = J_ir + J_rg;

%% calculate gradient

delta3 = zeros(m, num_labels);
delta2 = zeros(m, hidden_layer_size);
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for cnt = 1:m
    delta3(cnt,:) = a3_all(cnt,:) - all_labels(y(cnt),:);
    delta2(cnt,:) = delta3(cnt,:)*Theta2(:,2:end).*sigmoidGradient(z2_all(cnt,:));
    D1 = D1 + delta2(cnt,:)'*a1_all(cnt,:);
    D2 = D2 + delta3(cnt,:)'*a2_all(cnt,:);
end
Theta1_grad = 1/m*D1;
Theta2_grad = 1/m*D2;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
