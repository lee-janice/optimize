"""
Executes ISTA (Iterative Shrinkage Thresholding Algorithm)
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
"""
import numpy as np
import numpy.random as random
import numpy.linalg as la
import string
np.random.seed(0)

# importing functions from other files 
import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot
import generate_vectors as gen

def get_residual(A, x, y):
    """
    Calculates the residual (Ax-y)
    """
    residual = np.dot(A, x) - y
    return residual

def get_gradient(A, residual):
    """
    Calculates the gradient (A.T * (Ax-y))
    """
    gradient = np.dot(A.T, residual)
    return gradient

def ista(m, n, num_samp, max_iter, sparse=True, noise=False):
    """
    Executes ISTA 
    params:
        m (int):
        n (int):
        num_samp (int):
        max_iter (int):
    returns:
        results (array-like): an array containing 
    """
    # ------ SETTING PARAMETERS ------
    # true value of x (x*, solution)
    if (sparse):
        x_true = gen.rand_sparse(n, 50)
    else:
        x_true = gen.rand_exp_decay(n, 0.0001, np.sqrt(5))

    # true values of A and y
    A = random.randn(m, n)
    y_true = np.dot(A, x_true)

    if (noise):
        y = y_true + random.normal(0, 1, y_true.shape)
        #TODO: FIX THIS LOL 
        # y = gen.add_awgn_noise(y_true, -20)
    else:
        y = y_true

    # current values of x and z
    # column 0: classic, column 1: modified, column 2: modified + no threshold
    x_lb = np.zeros((n, 1), dtype=float)
    # x_lb = np.repeat(gen.rand_exp_decay(n, 0.0001, np.sqrt(5)), 3)
    # x_lb = np.reshape(x_lb, (n, 3))
    z_lb = np.zeros((n, 1), dtype=float)
    t_k_old = np.zeros((max_iter, 1), dtype=float)

    # threshold parameter
    lambda_lb = 4.0
    m_flag = np.zeros((1, n), dtype=int)
    
    # thresholding function 
    S = lambda x, lmda: np.multiply(np.maximum(np.absolute(x) - lmda, 0), np.sign(x))
    
    # arrays to hold results 
    residual = np.zeros((max_iter, 1), dtype=float)
    onenorm = np.zeros((max_iter, 1), dtype=float)
    moder = np.zeros((max_iter, 1), dtype=float)    # model error 

    # ------ MAIN LOOP ------
    for i in range(1, max_iter+1):
        print("iteration: " + str(i))
        
        # ------ SAMPLING ------
        # choosing random rows of A
        idx = random.permutation(n)

        # getting the corresponding rows of A and y
        A_sub = A[idx[:num_samp], :]
        y_sub = y[idx[:num_samp]]
        # TODO: FIX THIS LOL 
        # y_sub = y[0, idx[:num_samp]]

        # why is this here?
        t_lb = np.repeat(1/la.norm(A_sub, 2), 1)

        # ------ RESIDUAL AND GRADIENT ------
        # get residual
        r_lb = np.zeros((num_samp, 1))
        r_lb = get_residual(A_sub, x_lb, y_sub.reshape(num_samp, 1))

        # getting gradient
        g_lb = np.zeros((n, 1))
        g_lb = get_gradient(A_sub, r_lb)

        # ------ STEP SIZE ------
        # getting the step size
        t_lb = la.norm(r_lb[:, 0], 2)**2/la.norm(g_lb, 2)**2
        t_k_old[i-1, :] = t_lb
        
        # ------ CALCULATING X AND Z  ------
        z_lb = x_lb - t_lb*g_lb
        # calculating x_(k+1)
        x_lb = S(z_lb, lambda_lb*t_lb)
        
        # ------ RESULTS ------
        residual[i-1] = la.norm(get_residual(A_sub, x_lb, y_sub), 2) / la.norm(y_sub, 2)
        onenorm[i-1] = la.norm(x_lb, 1)
        moder[i-1] = la.norm(x_true - x_lb, 2) / la.norm(x_true, 2)
        
    return residual, onenorm, moder

def main():
    # ------ CONFIGURE PARAMETERS ------
    m = 2000         # rows of A 
    n = 100          # columns of A (rows of x_true and y_true)
    num_samp = 20    # rows of A and y to sample, num_samp < n
    max_iter = 250
    sparse = True
    noise = False
    
    plot_residual = True
    plot_onenorm = True
    plot_moder = True 
    # ------ EXECUTE ------
    results = ista(m, n, num_samp, max_iter, sparse, noise)
    
    print(results[0][:,0])
    # print(results[1])
    # print(results[2])
    
    if (plot_residual):
        plot.plot_ista(max_iter, results[0], sparse, noise, "residual")
    if (plot_onenorm):
        plot.plot_ista(max_iter, results[1], sparse, noise, "1-norm")
    if (plot_moder):
        plot.plot_ista(max_iter, results[2], sparse, noise, "model-error")
        
if __name__ == "__main__":
    main()    
