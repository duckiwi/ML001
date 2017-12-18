function p = predict(nn_params, X)
%   Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Initialize some useful variables
m = size(X, 1);
input_layer_size = layer_size(1);
hidden_layer_size = layer_size(2);
num_labels = layer_size(3);

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1_all = [ ones(m, 1), X];
z2_all = (Theta1*a1_all')';
a2_all = [ ones(size(z2_all, 1), 1), sigmoid(z2_all)];
z3_all = (Theta2*a2_all')';
a3_all = sigmoid(z3_all);

[val, idx] = max(a3_all, [], 2);
p = idx;

% =========================================================================

end
