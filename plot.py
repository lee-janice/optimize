"""
Plots the results of the methods and saves them as figures
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
"""
import matplotlib.pyplot as plt
import os 

def get_title(sparse, noise, type, algorithm):
    """
    Generates figure/plot titles
    params: 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
        algorithm (str): the type of algorithm used to obtain the data 
    returns:
        the generated figure/plot title 
    """
    title = algorithm.upper() + " ("
    # get sparse title 
    if (sparse): 
        title += "sparse soln, "
    else:
        title += "exponentally decaying soln, "
    # get noise title 
    if (noise):
        title += "w/noise) - "
    else: 
        title += "w/out noise) - "
    title += type.upper()
    return title 

def get_path(sparse, noise, type, algorithm):
    """
    Generates the path to save the figure/plots 
    params: 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
        algorithm (str): the type of algorithm used to obtain the data 
    returns:
        the generated figure/plot filename 
    """
    if not os.path.exists("plots/"):
        os.mkdir("plots/")
    
    if not os.path.exists("plots/" + algorithm + "/"):
        os.mkdir("plots/" + algorithm + "/")
        
    filename = "plots/" + algorithm + "/" + type
    # get sparse filename 
    if (sparse): 
        filename += "_sparse_"
    else:
        filename += "_randExpDecay_"
    # get noise filename 
    if (noise):
        filename += "noise.png"
    else: 
        filename += "noNoise.png"
    return filename
    
def plot_residual(max_iter, residual, sparse, noise, algorithm): 
    """
    Plots number of iterations vs. residual 
        params:
        max_iter (int): the number of iterations of the algorithm executed
        residual (array-like): the residual values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
    returns: 
        none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), residual)
    plt.xlabel("Number of iterations")
    plt.ylabel("Residual " + r"$||Ax-b||_{2}$")
    plt.title(get_title(sparse, noise, "residual", algorithm))
    plt.savefig(get_path(sparse, noise, "residual", algorithm))

def plot_moder(max_iter, moder, sparse, noise, algorithm): 
    """
    Plots number of iterations vs. model error 
    params:
        max_iter (int): the number of iterations of the algorithm executed
        moder (array-like): the model error values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm, model error, residual-vs-sparsity)
    returns: 
    none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), moder)
    plt.xlabel("Number of iterations")
    plt.ylabel("Model error " + r"$\frac{||x^*-x||_{2}}{||x^*||_{2}}$")
    plt.title(get_title(sparse, noise, "model-error", algorithm))
    plt.savefig(get_path(sparse, noise, "model-error", algorithm))
    
def plot_onenorm(max_iter, onenorm, sparse, noise, algorithm): 
    """
    Plots number of iterations vs. 1-norm 
    params:
        max_iter (int): the number of iterations of the algorithm executed
        onenorm (array-like): the 1-norm values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm, model error, residual-vs-sparsity)
    returns: 
    none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), onenorm)
    plt.xlabel("Number of iterations")
    plt.ylabel("1-norm " + r"$||x||_{1}$")
    plt.title(get_title(sparse, noise, "1-norm", algorithm))
    plt.savefig(get_path(sparse, noise, "1-norm", algorithm))
    
def plot_residual_vs_sparsity(onenorm, residual, sparse, noise, algorithm): 
    """
    Plots 1-norm vs. residual 
    params:
        onenorm (array-like): the 1-norm values to be plotted 
        residual (array-like): the residual values to be plotted 
        moder (array-like): the model error values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm, model error, residual-vs-sparsity)
    returns: 
    none 
    """
    plt.clf()
    plt.plot(onenorm, residual)
    plt.xlabel("1-norm " + r"$||x||_{1}$")
    plt.ylabel("Residual " + r"$||Ax-b||_{2}$")
    plt.title(get_title(sparse, noise, "residual-vs-sparsity", algorithm))
    plt.savefig(get_path(sparse, noise, "residual-vs-sparsity", algorithm))
    
    
