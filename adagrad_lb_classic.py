"""
Executes ADAGRAD (Adaptive Gradient Descent) with Linearized Bregman-like thresholding 
@authors: Jimmy Singh and Janice Lee
@date: June 23rd, 2019
"""
import numpy as np
import numpy.random as random
import numpy.linalg as la

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

def adagrad_lb_classic(params):
    """
    Executes ADAGRAD  
    params:
        params (Params object): contains parameters for optimization 
    returns:
        results (array-like): a tuple containing the arrays 
                with the results of the optimization
    """
    # ------ PARAMETERS ------
    m = params.m         
    n = params.n         
    num_samp = params.num_samp    
    max_iter = params.max_iter
    
    lmbda = params.lmbda 
    eta = params.eta
    epsilon = params.epsilon
    
    sparse = params.sparse 
    noise = params.noise
    # -----------------------
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
        results.update_x_history(x_k, n)
        results.update_z_history(z_k, n)
        
    return results
    
def main(): 
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    # ------ EXECUTE ------
    results = adagrad_lb_classic(params)
    # ------ PLOT ------
    algorithm = "adagrad-lb-classic"
    plt = plot.Plot(params)
    plt.update_algorithm(algorithm, results, thresholding=True)
    plt.plot_all()
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    
