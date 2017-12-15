function [J, grad] = linRegCostFunc(X, y, theta)
%%  COMPUTECOST Compute cost for linear regression with regularization
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% TODO check regularization term and grad

% Initialize some useful values
m = length(y); % number of training examples

% Initialize output values
J = 0;
grad = zeros(size(theta));

%% Calculate cost function and gradient 
% Calculate cost function with regularization, vectorized form
s = 0;
for cnt = 1:m
    temp = (theta'*X(cnt,:)' - y(cnt)).^2;
    s = s + temp;
end

J = 1/2/m*s;
% TODO add grad term

end
