clc
close all

rng(4)
%% settings
addpath('./functions'); 
addpath('./Measures');
addpath('./data');

Dname = 'HW.mat';
load(Dname);

% data preprocess
num_views = length(X);
n = length(Y);

anchor_rate=0.1:0.1:1;

c = length(unique(Y));  
opt1. style = 4;
opt1. IterMax =50;
opt1. toy = 0;
opt1. k = 10;
t1=clock;[P1, alpha, y] = FastmultiCLR(X, c, anchor_rate(5), opt1, 10);t2=clock;
result = ClusteringMeasure_new(Y, y);
time = etime(t2,t1);





