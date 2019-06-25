"""
Executes ADAGRAD (Adaptive Gradient Descent) with Linearized Bregman-like thresholding 
@authors: Jimmy Singh and Janice Lee
@date: June 23rd, 2019
"""
import numpy as np
import numpy.random as random
import numpy.linalg as la
np.random.seed(0)

import set_params
import init_problem as init
import get_results 
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
    
    # creates a Results object to hold and update results 
    results = get_results.Results(max_iter, n, x_true)
    
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
        results.update_iteration()
        results.update_residuals(residual, b_sub)
        results.update_onenorm(x_k)
        results.update_moder(x_true, x_k)
        results.update_z_history(z_k, n)
        
    return results
    
def main(): 
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    m = params.m         
    n = params.n         
    num_samp = params.num_samp    
    max_iter = params.max_iter
    lmbda = params.lmbda 
    eta = params.eta
    epsilon = params.epsilon
    
    sparse = params.sparse 
    noise = params.noise
    # ------ EXECUTE ------
    results = adagrad_lb_classic(m, n, num_samp, max_iter, lmbda, sparse, noise, eta, epsilon)
    
    # print(results[0][299])
    # print(results[1])
    # print(results[2])
    
    algorithm = "adagrad-lb-classic"
    
    plot.meta_plot(max_iter, results, sparse, noise, algorithm, m, num_samp, lmbda)
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    