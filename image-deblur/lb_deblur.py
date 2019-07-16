import pylops
import numpy as np

def threshold(x, lmbda):
    return np.maximum(np.absolute(x) - lmbda, 0) * np.sign(x)
    
def lb(A, b, max_iter, num_samp, lmbda, type):
    x_k = np.zeros(op.shape[1])
    z_k = np.zeros(op.shape[1])
    
    for i in range(1, max_iter+1):
        idx = np.random.permutation(op.shape[1])
        op_sub = pylops.signalprocessing.Convolve2D(num_samp * op.shape[1], )
