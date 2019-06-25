"""
Executes ADAGRAD (Adaptive Gradient Descent) with (modified) Linearized Bregman-like thresholding 
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

def adagrad_lb_modified(m, n, num_samp, max_iter, lmbda, sparse=True, noise=False, eta=.5, epsilon=1e-6, flipping=False):
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
        flipping (bool): true if step sizes should only be updated when values cross threshold
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
    # step sizes (component-wise array)
    tau = np.zeros((n, 1))
    # thresholder array 
    z_k = np.zeros((n, 1))
    # the estimation of x_true 
    x_k = np.zeros((n, 1))
    
    if (flipping):
        # will be used to flag the indices in z_k to apply new step size rule to 
        m_flag = np.zeros((1, n), dtype=int)
    
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
        
        # ------ THRESHOLDING ------
        if (flipping):
            # finding the indices in m_flag that are zero
            ind_flag = np.argwhere(m_flag == 0)[:, 1]
            # finding the indices in the second column (modified) of z_lb 
            # that are greater than the threshold
            ind_c = np.argwhere(np.absolute(z_k[ind_flag]) > lmbda)
            # flagging indices that are above the threshold
            # m_flag[ind_flag[ind_c]] = 1
            m_flag[0, ind_c] = 1
            
            # eliminate flipping depending on flag 
            ind_elim = np.argwhere(m_flag == 1)[:, 1]
            ind_nelim = np.argwhere(m_flag == 0)[:, 1]
        
        # ------ STEP SIZE ------
        s_k = s_k + np.multiply(gradient, gradient)
        t_k = eta / np.sqrt(s_k + epsilon) 
        tau = tau + np.sign(-gradient)
        if (flipping):
            step_size_elim = (t_k[ind_elim] * np.absolute(tau[ind_elim])/i)
        else: 
            step_size = (t_k * np.absolute(tau)/i)
        
        # ------ UPDATING X AND Z------
        if (flipping):
            z_k[ind_elim] = z_k[ind_elim] - np.multiply(step_size_elim, gradient[ind_elim])
            z_k[ind_nelim] = z_k[ind_nelim] - t_k[ind_nelim] * gradient[ind_nelim]
            x_k = threshold(z_k, lmbda)
        else: 
            z_k = z_k - np.multiply(step_size, gradient)
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
    flipping = params.flipping
    # ------ EXECUTE ------
    results = adagrad_lb_modified(m, n, num_samp, max_iter, lmbda, sparse, noise, eta, epsilon, flipping)
    
    # print(results[0][299])
    # print(results[1])
    # print(results[2])
    
    if (flipping):
        algorithm = "adagrad-lb-modified-w-flipping"
    else: 
        algorithm = "adagrad-lb-modified"
    
    plot.meta_plot(max_iter, results, sparse, noise, algorithm, m, num_samp, lmbda)
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    
