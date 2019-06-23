"""
Executes ADAGRAD (Adaptive Gradient Descent)
@authors: Jimmy Singh and Janice Lee
@date: June 21th, 2019
"""
import numpy as np
import numpy.random as random
import numpy.linalg as la
import matplotlib.pyplot as plt 
np.random.seed(0)

import init_problem as init 
import plot

def adagrad(m, n, num_samp, max_iter, sparse=True, noise=False, eta=2, epsilon=1e-6):
    """
    """
    # initializes the Ax = b problem 
    problem = init.init_l1(m, n, num_samp, max_iter, sparse, noise)
    A = problem[0]
    x_true = problem[1]
    b = problem[2]
    
    # the cumulative sum of the squared gradient 
    s_k = np.zeros((n, 1))
    # the step size ( eta/sqrt(s_k + epsilon) )
    t_k = np.zeros((n, 1))
    # the estimation of x_true 
    x_k = np.zeros((n, 1))
    
    # stores the results for each iteration 
    residuals = np.zeros((max_iter))
    onenorm = np.zeros((max_iter))
    moder = np.zeros((max_iter))
    
    for i in range(1, max_iter+1):
        print("iteration: " + str(i))
        
        # ------ SAMPLING ------
        # chooses random rows
        idx = random.permutation(n)
        # gets corresponding rows of A and b 
        A_sub = A[idx[:num_samp], :]
        b_sub = b[idx[:num_samp]]
        
        # ------ RESIDUAL AND GRADIENT ------
        # gets the residual ( Ax - b )
        residual = np.dot(A_sub, x_k) - b_sub
        # gets the gradient ( A.T * residual )
        gradient = np.dot(A_sub.T, residual)
        
        # ------ STEP SIZE ------
        s_k = s_k + np.multiply(gradient, gradient)
        t_k = eta / np.sqrt(s_k + epsilon) 
        
        # ------ UPDATE X ------
        x_k = x_k - np.multiply(t_k, gradient)
        
        # ------ RESULTS ------
        residuals[i-1] = la.norm(residual, 2) / la.norm(b_sub, 2)
        onenorm[i-1] = la.norm(x_k, 1)
        moder[i-1] = la.norm(x_true - x_k, 2) / la.norm(x_true, 2)
    
    return residuals, onenorm, moder 
    
def main(): 
    # ------ CONFIGURE PARAMETERS ------
    m = 20000         # rows of A 
    n = 1000          # columns of A (rows of x_true and y_true)
    num_samp = 200    # rows of A and y to sample, num_samp < n
    max_iter = 250
    sparse = True
    noise = False
    eta = 1
    epsilon = 1e-6
    
    plot_residual = True
    plot_onenorm = True
    plot_moder = True 
    # ------ EXECUTE ------
    results = adagrad(m, n, num_samp, max_iter, sparse, noise, eta, epsilon)
    
    print(results[0][249])
    # print(results[1])
    # print(results[2])
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    
