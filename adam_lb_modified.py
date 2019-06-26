"""
Executes ADAM (Adaptive Moment Estimation) with Linearized Bregman-like thresholding 
@authors: Jimmy Singh and Janice Lee
@date: June 25th, 2019
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

def adam_lb_modified(params):
    """
    Executes ADAM with modified Linearized Bregman thresholding   
    params:
        params (Params object): contains parameters for optimization
    returns:
        results (Results object): contains the arrays 
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
    beta_1 = params.beta_1
    beta_2 = params.beta_2
    
    sparse = params.sparse 
    noise = params.noise
    flipping = params.flipping
    # -----------------------
    # initializes the Ax = b problem 
    problem = init.init_l1(m, n, num_samp, max_iter, sparse, noise)
    A = problem[0]
    x_true = problem[1]
    b = problem[2]
    
    # step sizes (component-wise array)
    tau = np.zeros((n, 1))
    # thresholder array 
    z_k = np.zeros((n, 1))
    # the estimation of x_true 
    x_k = np.zeros((n, 1))
    # the exponentially moving average of the gradient mean and variance
    m_k = np.zeros((n, 1))
    v_k = np.zeros((n, 1))
    
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
        
        # ------ FLIPPING ------
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
        
        # ------ UPDATING M AND V ------
        m_k = beta_1 * m_k + (1-beta_1) * gradient
        v_k = beta_2 * v_k + (1-beta_2) * gradient**2
        # bias correction 
        m_hat = m_k / (1-beta_1**i) 
        v_hat = v_k / (1-beta_2**i) 
        
        # ------ STEP SIZE ------
        t_k = eta / (np.sqrt(v_hat) + epsilon)
        tau = tau + np.sign(-gradient)
        if (flipping):
            step_size_elim = (t_k[ind_elim] * np.absolute(tau[ind_elim])/i)
        else: 
            step_size = (t_k * np.absolute(tau)/i)
        
        # ------ UPDATING X AND Z------
        if (flipping):
            z_k[ind_elim] = z_k[ind_elim] - np.multiply(step_size_elim, m_hat[ind_elim])
            z_k[ind_nelim] = z_k[ind_nelim] - t_k[ind_nelim] * m_hat[ind_nelim]
            x_k = threshold(z_k, lmbda)
        else: 
            z_k = z_k - np.multiply(step_size, m_hat)
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
    results = adam_lb_modified(params)
    # ------ PLOT ------
    if (params.flipping):
        algorithm = "adam-lb-modified-w-flipping"
    else: 
        algorithm = "adam-lb-modified"
    plt = plot.Plot(params)
    plt.update_algorithm(algorithm, results, thresholding=True)
    plt.plot_all()
        
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    
