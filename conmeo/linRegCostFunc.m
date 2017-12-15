function [J, grad] = linRegCostFunc(X, y, theta, lambda)
%%  Computes the cost of using theta as the parameter for linear regression
%   to fit the data points in X and y. Returns the cost in J and the 
%   gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Initialize output values
J = 0;
grad = zeros(size(theta));

%% Calculate cost function and gradient 
% Calculate cost function with regularization, vectorized form
J = 1/2/m*sum((X*theta - y).^2) + lambda/2/m*sum(theta(2:end).^2);

% Calculate cost function with regularization, vectorized form
grad = 1/m*((X*theta - y)'*X)';
grad = grad + [0; lambda/m*theta(2:end)];

%% Transform grad into vector form
grad = grad(:);

end
