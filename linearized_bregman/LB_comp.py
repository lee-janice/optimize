"""
Compares classic Linearized Bregman to modified Linearized Bregman
@authors: Jimmy Singh and Janice Lee
@date: June 6th, 2019
"""
import numpy as np

def rand_exp_decay(n, a, b):
"""
???
params:
    n (int): desired number of elements in the vector
    a (float): ???
    b (float): ???
returns:
    none
"""
    A = log(a)
    B = log(b)
    R1 = A + (B-A) * np.rand(n)
    R = np.square(np.exp(R1))
    idx = np.random.permutation(n)
