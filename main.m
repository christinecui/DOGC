clc;
clear all;

% load dataset
load('vote_uni.mat');

% parameter setting
param.k = 10;
param.d = 3;
param.alpha = 0.00001;
param.beta = 0.0001;
param.gamma = 0.00001;
param.p = 1;
param.r = -1;

% iterations
iterations = 100;

for j = 1:iterations
    [W, S, Q, Final_Y, F, B, D, obj] = DOGC(X',Y, param);
    result = ClusteringMeasure(Y, Final_Y); % ACC,NMI,Purity 
end