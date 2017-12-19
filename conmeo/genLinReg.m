function out1 = genLinReg(X, y)
%% generate linear regression model parameters
X = [ones(m, 1), X]; % Add a column of ones to X
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);
