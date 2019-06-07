% clea% Sparse recovery
% Code compares Classic LB to  modified LB
for ii = 1:3

%clear;% clc; %close all;
rng(6)


%Toy Dimensions
dToy = 2000;
m = dToy;
dToySol = 100;
n = dToySol;
%create compressible vector
xTrue = rand_exp_decay(n,0.0001,sqrt(5));


A = randn(length(1:dToy),length(1:dToySol));

yTrue = A*xTrue;

y = add_awgn_noise(yTrue, -20);
%y = yTrue;
sigma = norm(yTrue-y,2);

noise = yTrue - y;
%A = awgn(A, 20);

S = @(x,lambda) max(abs(x)-lambda,0).*sign(x);
Sp = @(x,lambda) max(abs(1)-lambda,0).*sign(1);


%% Compare solvers: set up parameters

nrows = 250;
maxIter = 250;
iter=[1:maxIter];
x_lb = zeros(n,3);
z_lb = zeros(n,3);

Store1=zeros(n,maxIter);
Store2=zeros(n,maxIter);
Store3=zeros(n,maxIter);

t_k_new=zeros(n,1);
t_k_new2=zeros(n,1);

lambda_lb = 4.0;
mflag = zeros(dToySol,1);
%% main loop
for j=1:maxIter
    disp(j)
    % choose random rows of A
    idx = randperm(m);

    Asub = A(idx(1:nrows),:);
    ysub = transpose(y(idx(1:nrows)));

    t_lb([1 2 3]) = 1/norm(Asub,2)^2;



    % Bregman
    %full matrix residual and gradient

    r_lb(:,1) = Asub*x_lb(:,1) - ysub;
    r_lb(:,2) = Asub*x_lb(:,2) - ysub;
    r_lb(:,3) = Asub*x_lb(:,3) - ysub;

    g_lb(:,1) = Asub'*r_lb(:,1);
    g_lb(:,2) = Asub'*r_lb(:,2);
    g_lb(:,3) = Asub'*r_lb(:,3);


    t_lb(1) = norm(r_lb(:,1))^2/norm(g_lb(:,1))^2;
    t_lb(2) = norm(r_lb(:,2))^2/norm(g_lb(:,2))^2;
    t_lb(3) = norm(r_lb(:,3))^2/norm(g_lb(:,3))^2;
