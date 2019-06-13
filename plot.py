"""
Plots the results of the methods and saves them as figures
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
"""
import matplotlib.pyplot as plt
import os 

def get_title(sparse, noise, type):
    """
    Generates figure/plot titles
    params: 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
    returns:
        the generated figure/plot title 
    """
    title = "("
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

def get_filename(sparse, noise, type):
    """
    Generates figure/plot filenames 
    params: 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
    returns:
        the generated figure/plot filename 
    """
    if not os.path.exists("plots/"):
        os.mkdir("plots/")
        
    filename = "plots/" + type
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

def plot_lb(max_iter, data, sparse, noise, type):
    """
    Plots the data and saves the plot
    params:
        max_iter (int): the number of iterations of the algorithm executed
        data (array-like): the data values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
    returns: 
        none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), data)
    plt.legend(["Classic", "Modified", "Modified w/out threshold"])
    plt.xlabel("Number of iterations")
    plt.ylabel("Data values")
    plt.title("LB " + get_title(sparse, noise, type))
    plt.savefig(get_filename(sparse, noise, type))
    
def plot_ista(max_iter, data, sparse, noise, type):
    """
    Plots the data and saves the plot
    params:
        max_iter (int): the number of iterations of the algorithm executed
        data (array-like): the data values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
        type (str): the type of data to be plotted (residual, 1-norm,  model error)
    returns: 
        none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), data)
    plt.xlabel("Number of iterations")
    plt.ylabel("Data values")
    plt.title("ISTA " + get_title(sparse, noise, type))
    plt.savefig(get_filename(sparse, noise, type))
