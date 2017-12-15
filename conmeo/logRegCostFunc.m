function [J, grad] = logRegCostFunc(theta, X, y, lambda)
%%  Computes the cost of using theta as the parameter for logistic regression
%   to fit the data points in X and y. Returns the cost in J and the 
%   gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Initialize output values
J = 0;
grad = zeros(size(theta));

%% Calculate cost function and gradient 
% Calculate cost function with regularization, vectorized form
J = 1/m*sum(-y.*log(sigmoid(X*theta)) - (1 - y).*log(1 - sigmoid(X*theta))) + lambda/2/m*sum(theta(2:end).^2);

% Calculate gradient with regularization, vectorized form
grad = 1/m*X'*(sigmoid(X*theta) - y);
temp = theta;
temp(1) = 0;
grad = grad + lambda/m*temp;

%% Transform grad into vector form
grad = grad(:);

end
