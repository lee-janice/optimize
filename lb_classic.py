"""
Executes the classic Linearized Bregman 
@authors: Jimmy Singh and Janice Lee
@date: June 6th, 2019
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
    
def lb_classic(params):
    """
    Executes classic Linearized Bregman  
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
    
    sparse = params.sparse 
    noise = params.noise
    # ------------------------
    # initializes the Ax = y problem 
    problem = init.init_l1(m, n, num_samp, max_iter, sparse, noise)
    A = problem[0]
    x_true = problem[1]
    b = problem[2]
    
    # current values of x and z 
    x_k = np.zeros((n, 1))
    z_k = np.zeros((n, 1))

    # creates a Results object to hold and update results 
    results = get_results.Results(max_iter, n, x_true)
    
    # ------ MAIN LOOP ------
    for i in range(1, max_iter+1):
        print("iteration: " + str(i))
        
        # ------ SAMPLING ------
        # choosing random rows of A
        idx = random.permutation(n)

        # getting the corresponding rows of A and y
        A_sub = A[idx[:num_samp], :]
        b_sub = b[idx[:num_samp]]
        # TODO: FIX THIS LOL 
        # y_sub = y[0, idx[:num_samp]]

        # ------ RESIDUAL AND GRADIENT ------
        # gets the residual ( Ax - b )
        residual = np.dot(A_sub, x_k) - b_sub
        # gets the gradient ( A.T * residual )
        gradient = np.dot(A_sub.T, residual)
        
        # ------ STEP SIZE ------
        # getting the step size
        t_k = la.norm(residual, 2)**2/la.norm(gradient, 2)**2
        
        # ------ UPDATING X AND Z  ------
        z_k = z_k - t_k * gradient
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
    results = lb_classic(params)
    # ------ PLOT ------
    algorithm = "lb-classic"
    plt = plot.Plot(params)
    plt.update_algorithm(algorithm, results, thresholding=True)
    plt.plot_all()
        
        
if __name__ == "__main__":
    main()    
        
