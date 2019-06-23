function R = rand_exp_decay(n,a,b)

A = log(a);
B = log(b);
R1 = A + (B-A).*rand(n,1);
R = exp(R1).^2;
idx = randperm(n);
%replaces half of the elements in R with their negative values (elements chosen randomly)
%NOTE: why/what does this function do? what are the parameters a and b? 
R(idx(1:floor(n/2))) = -R(idx(1:floor(n/2)));

end
