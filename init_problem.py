"""
Initializes the Ax = b minimization problem 
@authors: Jimmy Singh and Janice Lee
@date: June 21st, 2019 
"""
import numpy as np
import generate_vectors as gen

def init_l1(m, n, num_samp, max_iter, sparse=True, noise=False):
    """
    Initializes Ax = b for an l1-norm minimization/basis pursuit problem
    """
    # initializes the true value of x (x*)
    if (sparse):
        x_true = gen.rand_sparse(n, 50)
    else:
        x_true = gen.rand_exp_decay(n, 0.0001, np.sqrt(5))
        
    # initializes the true values of A and b
    # true values of A and y
    A = np.random.randn(m, n)
    b_true = np.dot(A, x_true)
    
    # adds noise if needed 
    if (noise):
        b = b_true + np.random.normal(0, 1, b_true.shape)
        #TODO: FIX THIS LOL 
        # y = gen.add_awgn_noise(y_true, -20)
    else:
        b = b_true
        
    return (A, x_true, b)
