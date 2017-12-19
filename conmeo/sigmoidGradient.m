function g = sigmoidGradient(z)
%%  Returns the gradient of the sigmoid function evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This work sregardless if z is a matrix or a vector. In 
%   particular, if z is a vector or matrix, return the gradient for each element.

g = sigmoid(z).*(1 - sigmoid(z));

end
