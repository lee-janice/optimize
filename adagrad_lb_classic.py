"""
Executes ADAGRAD (Adaptive Gradient Descent) with Linearized Bregman-like thresholding 
@authors: Jimmy Singh and Janice Lee
@date: June 23rd, 2019
"""
import numpy as np
import numpy.random as random
import numpy.linalg as la
np.random.seed(0)

import init_problem as init 
import plot

def threshold(x, lmbda): 
    """
    Replaces values in x that are less than lambda with 0
    params: 
        x (array-like): the array to threshold
        lmbda (float): the value to threshold by 
    returns: 
        a thresholded array 
    """
    return np.multiply(np.maximum(np.absolute(x) - lmbda, 0), np.sign(x))

def adagrad_lb_classic(m, n, num_samp, max_iter, lmbda, sparse=True, noise=False, eta=.5, epsilon=1e-6):
    """
    Executes ADAGRAD  
    params:
        m (int): rows of A 
        n (int): columns of A / rows of x and b
        num_samp (int): rows of A and b to sample, num_samp < n 
        max_iter (int): number of iterations to run 
        lmbda (float): the thresholding parameter 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        eta (float): the desired learning rate 
        epsilon (float): a small constant to avoid division by zero in calculation of step size 
    returns:
        results (array-like): a tuple containing the arrays 
                with the results of the optimization
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
    # thresholder array 
    z_k = np.zeros((n, 1))
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
        
        # ------ UPDATING X AND Z------
        z_k = z_k - np.multiply(t_k, gradient)
        x_k = threshold(z_k, lmbda)
        
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
    max_iter = 300
    sparse = True
    noise = False
    lmbda = 3.0
    eta = .5
    epsilon = 1e-6
    
    plot_residual = True
    plot_onenorm = True
    plot_moder = True 
    # ------ EXECUTE ------
    results = adagrad_lb_classic(m, n, num_samp, max_iter, lmbda, sparse, noise, eta, epsilon)
    
    print(results[0][299])
    # print(results[1])
    # print(results[2])
    
    algorithm = "adagrad-lb-classic"
    
    if (plot_residual):
        plot.plot_residual(max_iter, results[0], sparse, noise, algorithm)
    if (plot_onenorm):
        plot.plot_onenorm(max_iter, results[1], sparse, noise, algorithm)
    if (plot_moder):
        plot.plot_moder(max_iter, results[2], sparse, noise, algorithm)
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    
