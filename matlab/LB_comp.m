% Code compares Classic LB to  modified LB
%for ii = 1:3

%clear;% clc; %close all;
%rng(6)


%Toy Dimensions
dToy = 200;
m = dToy;
dToySol = 1000;
n = dToySol;
%create compressible vector
%xTrue = rand_exp_decay(n,0.0001,sqrt(5));
xTrue = zeros(n, 1);
idx = randperm(n);
elem = idx(1,1);

for ii= 1:floor(n/5)
xTrue(idx(ii)) = rand(1);
end

%xTrue(idx(1:floor(n/5))) = rand(floor(n/5),1);

A = randn(length(1:dToy),length(1:dToySol));

yTrue = A*xTrue;

%y = add_awgn_noise(yTrue, -20);
y = yTrue;
sigma = norm(yTrue-y,2);

%noise = yTrue - y;
%A = awgn(A, 20);

S = @(x,lambda) max(abs(x)-lambda,0).*sign(x);
Sp = @(x,lambda) max(abs(1)-lambda,0).*sign(1);


%% Compare solvers: set up parameters

nrows = 200;
maxIter = 1000;
iter=[1:maxIter];
x_lb = zeros(n,3);
z_lb = zeros(n,3);
elem_lb = zeros(250,3);

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
    ysub = y(idx(1:nrows));
    %ysub = transpose(ysub);

    t_lb([1 2 3]) = 1/norm(Asub,2)^2;



    % Bregman
    %full matrix residual and gradient

    r_lb = zeros(nrows, 3);
    r_lb(:,1) = Asub*x_lb(:,1) - ysub;
    r_lb(:,2) = Asub*x_lb(:,2) - ysub;
    r_lb(:,3) = Asub*x_lb(:,3) - ysub;

    g_lb = zeros(dToySol, 3);
    g_lb(:,1) = Asub'*r_lb(:,1);
    g_lb(:,2) = Asub'*r_lb(:,2);
    g_lb(:,3) = Asub'*r_lb(:,3);


    t_lb(1) = norm(r_lb(:,1))^2/norm(g_lb(:,1))^2;
    t_lb(2) = norm(r_lb(:,2))^2/norm(g_lb(:,2))^2;
    t_lb(3) = norm(r_lb(:,3))^2/norm(g_lb(:,3))^2;



    t_k_old(j,:)=t_lb;
    t_k_new = t_k_new+sign(-g_lb(:,2));
    t_k_new2 = t_k_new2+sign(-g_lb(:,3));



    %Flaging: flag everithing that goes above the threshold.
    % update flag
    %Looking at 2nd column(modified LB)
    ind_flag = find(mflag==0);
    ind_c = find(abs(z_lb(ind_flag,2)) > lambda_lb);
    mflag(ind_flag(ind_c)) = 1;

    % eliminate flipping depending on flag.
    ind_elim = find(mflag==1);
    ind_nelim = find(mflag==0);

    z_lb(ind_elim,2) = z_lb(ind_elim,2) - (t_lb(2)*abs(t_k_new(ind_elim))/j).*g_lb(ind_elim,2);
    z_lb(ind_nelim,2) = z_lb(ind_nelim,2) - (t_lb(2))*g_lb(ind_nelim,2);

    %No threshold detection
    z_lb(:,3) = z_lb(:,3) - (t_lb(3)*abs(t_k_new2)/j).*g_lb(:,3);
    %Classic
    z_lb(:,1) = z_lb(:,1) - t_lb(1)*g_lb(:,1);


    %x is the sparse soln & z is calc.the next value of z
    x_lb(:,1:3) = S(z_lb(:,1:3),lambda_lb);

    Store1(:,j)=z_lb(:,1);
    Store2(:,j)=z_lb(:,2);
    Store3(:,j)=z_lb(:,3);


    %Residual
    residual(j,1) = norm(Asub*x_lb(:,1) - ysub,2)/norm((ysub));
    residual(j,2) = norm(Asub*x_lb(:,2) - ysub,2)/norm((ysub));
    residual(j,3) = norm(Asub*x_lb(:,3) - ysub,2)/norm((ysub));


    % One norm
    onenorm(j,1) = norm(x_lb(:,1),1);
    onenorm(j,2) = norm(x_lb(:,2),1);
    onenorm(j,3) = norm(x_lb(:,3),1);

    % Model error
    %Slide #24 of snowb_19
    moder(j,1)=norm(xTrue-x_lb(:,1),2)/norm(xTrue,2);
    moder(j,2)=norm(xTrue-x_lb(:,2),2)/norm(xTrue,2);
    moder(j,3)=norm(xTrue-x_lb(:,3),2)/norm(xTrue,2);

    elem_lb(j,:) = x_lb(elem,:);
end
%plot(1:250,moder(:,:));
plot(1:maxIter,elem_lb(:,:));
%hold on;
%end
