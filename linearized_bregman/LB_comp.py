"""
Compares classic Linearized Bregman to modified Linearized Bregman
@authors: Jimmy Singh and Janice Lee
@date: June 6th, 2019
"""
import numpy as np

def rand_exp_decay(n, a, b):
    """
    Creates a random semi-sparse array with exponentially decreasing elements
    params:
        n (int): desired number of elements in the vector
        a (float): ???
        b (float): ???
    returns:
        R (numpy array): array with exponentially decreasing elements
    """
    A = np.log(a)
    B = np.log(b)
    R1 = A + (B-A) * np.random.rand(n)
    print(R1)
    R = np.exp(R1*2)
    print(R)
    idx = np.random.permutation(n)
    for i in idx[1:int(np.floor(n/2))]:
        R[i] = -R[i]
    return R

def rand_sparse(n, frac):
    """
    Creates a random sparse array
    """
    R = np.zeros(n, dtype=int)
    idx = np.random.permutation(n)
    for i in idx[1:(n/frac)]:
        R[i] = np.random.rand(n)
    return R


def lb_compare(m, n, num_samp, max_iter):
    """
    Compares classic LB to modified LB
    params:
        m (int):
        n (int):
        num_samp (int):
        max_iter (int):
    returns:
    """
