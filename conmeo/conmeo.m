%% sample program for machine learning demo
% the program will use data from coursera assignments
% the program includes:
% 

%       import data, extract information
%       plot data for visualization
%       preprocessing data, with feature scaling
%       linear regression with regulization
%       logistic regression with regulization
%       neural network with regulization, parameter randomization along with debug tools
%       plot result for visualization
%       cross validation and testing
%       analyze the role of regulization
%       analyze the size of training set by prediction accuracy/error and plot training curve

clear; close all; clc

%% ======================= Part 01: Import data and visualization =======================
% Input
file_name = 'ex4data1.mat';
% Load data
fprintf('Loading and Visualizing Data ...\n')
load(file_name);
% initialize important variables
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
fprintf('Program paused. Press enter to continue.\n');
pause;
clear sel

%% ======================= Part 02: Linear regression =======================
% Input
iteration = 1000;
thetaLinReg = genLinReg(X, y);




%% logistic regression

logRegResult = genLogReg(data_set);



%% neural network
% set up architecture




