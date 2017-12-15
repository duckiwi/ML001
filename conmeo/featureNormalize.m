function [X_norm, mu, sigma] = featureNormalize(X)
%%  Normalizes the features in X 
%   Returns a normalized version of X where the mean value of each feature
%   is 0 and the standard deviation is 1. 

%% Initialize input values
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

%% Calculate normalized feature
mu = mean(X);
for cnt = 1:size(X,2)
    X_norm(:,cnt) = X_norm(:,cnt) - mu(cnt);
end
sigma = std(X);
for cnt = 1:size(X,2)
    X_norm(:,cnt) = X_norm(:,cnt)/sigma(cnt);
end

end
