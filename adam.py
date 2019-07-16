"""
Executes ADAM (Adaptive Moment Estimation)
@authors: Jimmy Singh and Janice Lee
@date: June 25th, 2019
"""
import numpy as np
import numpy.random as random
import numpy.linalg as la
np.random.seed(0)

import set_params
import init_problem as init 
import get_results
import plot

def adam(params):
    """
    Executes ADAM  
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
    beta_1 = params.beta_1
    beta_2 = params.beta_2
    
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
    # the estimation of x_true 
    x_k = np.zeros((n, 1))
    # the exponentially moving average of the gradient mean and variance
    m_k = np.zeros((n, 1))
    v_k = np.zeros((n, 1))
    
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
        
        # ------ UPDATING M AND V ------
        m_k = beta_1 * m_k + (1-beta_1) * gradient
        v_k = beta_2 * v_k + (1-beta_2) * gradient**2
        # bias correction 
        m_hat = m_k / (1-beta_1**i) 
        v_hat = v_k / (1-beta_2**i) 
        
        # ------ STEP SIZE ------
        t_k = eta / (np.sqrt(v_hat) + epsilon)
        
        # ------ UPDATING X AND Z------
        x_k = x_k - np.multiply(t_k, m_hat)
        
        # ------ RESULTS ------
        results.update(residual, b_sub, n, x_k, np.zeros((n, 1)), t_k, adaptive=True)
        
        # print(t_k)
        # print(x_k[:20])
        
    return results
    
def main(): 
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    # ------ EXECUTE ------
    results = adam(params)
    # ------ PLOT ------
    algorithm = "adam"
    plt = plot.Plot(params)
    plt.update_algorithm(algorithm, results, thresholding=False)
    plt.plot_all()
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
    
    
    
    
