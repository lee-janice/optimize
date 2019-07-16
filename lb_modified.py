"""
Executes the modified Linearized Bregman (adaptive step size)
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
    
def lb_modified(params):
    """
    Executes modified Linearized Bregman  
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
    flipping = params.flipping
    # ------------------------
    # initializes the Ax = y problem 
    problem = init.init_l1(m, n, num_samp, max_iter, sparse, noise)
    A = problem[0]
    x_true = problem[1]
    b = problem[2]
    
    # current values of x and z 
    x_k = np.zeros((n, 1))
    z_k = np.zeros((n, 1))
    
    # step sizes (component-wise array)
    tau = np.zeros((n, 1))
    
    if (flipping):
        # will be used to flag the indices in z_k to apply new step size rule to 
        m_flag = np.zeros((1, n), dtype=int)

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
        t_k = la.norm(residual, 2)**2/la.norm(gradient, 2)**2
        # getting the component-wise update 
        tau = tau + np.sign(-gradient)
        # getting the step size 
        if (flipping):
            step_size = (t_k * np.absolute(tau[ind_elim])/i)
        else: 
            step_size = (t_k * np.absolute(tau)/i)
        
        # ------ UPDATING X AND Z ------
        if (flipping):
            z_k[ind_elim] = z_k[ind_elim] - np.multiply(step_size, gradient[ind_elim])
            z_k[ind_nelim] = z_k[ind_nelim] - t_k * gradient[ind_nelim]
            x_k = threshold(z_k, lmbda)
        else: 
            z_k = z_k - np.multiply(step_size, gradient)
            x_k = threshold(z_k, lmbda)
        
        # ------ RESULTS ------
        results.update(residual, b_sub, n, x_k, z_k, t_k, adaptive=False)
        
    return results

def main():
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    # ------ EXECUTE ------
    results = lb_modified(params)
    # ------ PLOT ------
    if (params.flipping):
        algorithm = "lb-modified-w-flipping"
    else: 
        algorithm = "lb-modified"
    plt = plot.Plot(params)
    plt.update_algorithm(algorithm, results, thresholding=True)
    plt.plot_all()
        
if __name__ == "__main__":
    main()    
