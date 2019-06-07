function R = rand_exp_decay(n,a,b)

A = log(a);
B = log(b);
R1 = A + (B-A).*rand(n,1);
R = exp(R1).^2;
idx = randperm(n);
R(idx(1:floor(n/2))) = -R(idx(1:floor(n/2)));
end