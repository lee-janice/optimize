"""
Generates vectors/arrays to use in optimization algorithms 
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
"""
import numpy as np
import numpy.random as random
np.random.seed(0)

def rand_exp_decay(n, a, b):
    """
    Creates a random semi-sparse array with exponentially decreasing elements
    params:
        n (int): desired number of elements in the array
        a (float): ???
        b (float): ???
    returns:
        R (numpy array): array with exponentially decreasing elements
    """
    A = np.log(a)
    B = np.log(b)
    R1 = A + (B-A) * random.rand(n)
    R = np.exp(R1*2)
    idx = random.permutation(n)
    for i in idx[0:int(np.floor(n/2))]:
        R[i] = -R[i]
    return R.reshape(n, 1)
    
def rand_sparse(n, frac):
    """
    Creates a random sparse array
    params:
        n (int): desired number of elements in the array
        frac (int): 1/frac gives the fraction of non-zero elements in the array
    returns:
        R (numpy array): sparse array
    """
    R = np.zeros((n,1), dtype=float)
    # R = np.zeros(n, dtype=float)
    idx = random.permutation(n)
    for i in idx[0:(n/frac)]:
        R[i] = random.rand(1)
    return R
    
def add_awgn_noise(x, snr_dB):
    """
    Function to add AWGN to a given signal 
    Original function authored by Mathuranathan Viswanathan for MatLab/Octave
    """
    l = x.size
    snr = 10 ** (snr_dB/10)
    e_sym = np.sum(np.square(x)) / l
    n0 = e_sym / snr
    if (np.isreal(x).all()):
        noise_sigma = np.sqrt(n0)
        n = noise_sigma * random.randn(1, l) 
    else:
        noise_sigma = np.sqrt(n0/2)
        n = noise_sigma * (random.randn(1, l) + 1j*random.randn(1, l))
    y = x + n
    return y
