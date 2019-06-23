"""
Executes ISTA (Iterative Shrinkage Thresholding Algorithm)
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
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

def ista(m, n, num_samp, max_iter, lmbda, sparse=True, noise=False):
    """
    Executes ISTA 
    params:
        m (int): rows of A 
        n (int): columns of A / rows of x and b
        num_samp (int): rows of A and b to sample, num_samp < n 
        max_iter (int): number of iterations to run 
        lmbda (float): the thresholding parameter 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns:
        results (array-like): an array containing the results of the optimization
    """
    # ------ SETTING PARAMETERS ------
    # initializes the Ax = y problem 
    problem = init.init_l1(m, n, num_samp, max_iter, sparse, noise)
    A = problem[0]
    x_true = problem[1]
    b = problem[2]

    # current values of x and z
    x_k = np.zeros((n, 1))
    z_k = np.zeros((n, 1))

    # arrays to hold results 
    residuals = np.zeros((max_iter))
    onenorm = np.zeros((max_iter))
    moder = np.zeros((max_iter)) 

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
        z_k = x_k - t_k * gradient
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
    max_iter = 1000
    lmbda = 3.0
    sparse = True
    noise = False
    
    plot_residual = True
    plot_onenorm = True
    plot_moder = True 
    # ------ EXECUTE ------
    results = ista(m, n, num_samp, max_iter, lmbda, sparse, noise)
    
    # print(results[0])
    # print(results[1])
    # print(results[2])
    
    if (plot_residual):
        plot.plot_residual(max_iter, results[0], sparse, noise, "ISTA")
    if (plot_onenorm):
        plot.plot_onenorm(max_iter, results[1], sparse, noise, "ISTA")
    if (plot_moder):
        plot.plot_moder(max_iter, results[2], sparse, noise, "ISTA")
        
if __name__ == "__main__":
    main()    
