"""
Plots the results of the methods and saves them as figures
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
"""
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
plt.style.use('ggplot')
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
    if (sparse): 
        subclass = "sparse_"
    else:
        subclass = "randExpDecay_"
    if (noise):
        subclass += "noise/"
    else: 
        subclass += "noNoise/"
        
    if not os.path.exists("plots/"):
        os.mkdir("plots/")
    
    if not os.path.exists("plots/" + algorithm + "/"):
        os.mkdir("plots/" + algorithm + "/")
    
    if not os.path.exists("plots/" + algorithm + "/" + subclass):
        os.mkdir("plots/" + algorithm + "/" + subclass)
        
    filename = "plots/" + algorithm + "/" + subclass + type
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
    
def plot_residual(max_iter, residual, sparse, noise, m, num_samp, legend=None): 
    """
    Plots number of iterations vs. residual and saves the plot
        params:
        max_iter (int): the number of iterations of the algorithm executed
        residual (array-like): the residual values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns: 
        none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), residual)
    add_datapass_indicator(max_iter, m, num_samp)
    
    plt.xlabel("Number of iterations")
    plt.ylabel("Residual " + r"$||Ax-b||_{2}$")
    save_plot(sparse, noise, type, algorithm)

def plot_moder(max_iter, moder, sparse, noise, m, num_samp, legend=None): 
    """
    Plots number of iterations vs. model error and saves the plot
    params:
        max_iter (int): the number of iterations of the algorithm executed
        moder (array-like): the model error values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns: 
    none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), moder)
    add_datapass_indicator(max_iter, m, num_samp)
    
    plt.xlabel("Number of iterations")
    plt.ylabel("Model error " + r"$\frac{||x^*-x||_{2}}{||x^*||_{2}}$")
    save_plot(sparse, noise, type, algorithm)
    
def plot_onenorm(max_iter, onenorm, sparse, noise, m, num_samp, legend=None): 
    """
    Plots number of iterations vs. 1-norm and saves the plot
    params:
        max_iter (int): the number of iterations of the algorithm executed
        onenorm (array-like): the 1-norm values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns: 
    none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), onenorm)
    add_datapass_indicator(max_iter, m, num_samp)
    
    plt.xlabel("Number of iterations")
    plt.ylabel("1-norm " + r"$||x||_{1}$")
    save_plot(sparse, noise, type, algorithm)
    
def plot_residual_vs_sparsity(onenorm, residual, sparse, noise, m, num_samp, legend=None): 
    """
    Plots 1-norm vs. residual and saves the plot
    params:
        onenorm (array-like): the 1-norm values to be plotted 
        residual (array-like): the residual values to be plotted 
        moder (array-like): the model error values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns: 
    none 
    """
    plt.clf()
    plt.plot(onenorm, residual)
    add_datapass_indicator(max_iter, m, num_samp)
    
    plt.xlabel("1-norm " + r"$||x||_{1}$")
    plt.ylabel("Residual " + r"$||Ax-b||_{2}$")
    save_plot(sparse, noise, type, algorithm)
    
def plot_x_nonzeros(max_iter, x_k, sparse, noise):
    """
    Plots progression of z_k values and saves the plot 
    params:
        onenorm (array-like): the 1-norm values to be plotted 
        residual (array-like): the residual values to be plotted 
        moder (array-like): the model error values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns: 
    none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), x_k)
    
    plt.xlabel("Number of iterations")
    plt.ylabel(r"$x_k$")
    save_plot(sparse, noise, type, algorithm)

def plot_z_nonzeros(max_iter, z_k, lmbda, sparse, noise):
    """
    Plots progression of z_k values and saves the plot 
    params:
        onenorm (array-like): the 1-norm values to be plotted 
        residual (array-like): the residual values to be plotted 
        moder (array-like): the model error values to be plotted 
        sparse (bool): true if the soln is sparse 
        noise (bool): true if the data contains noise 
    returns: 
    none 
    """
    plt.clf()
    plt.plot(range(1, max_iter+1), z_k)
    
    plt.axhline(y=lmbda, linestyle="--")
    plt.axhline(y=-lmbda, linestyle="--")
    
    plt.xlabel("Number of iterations")
    plt.ylabel(r"$z_k$")
    save_plot(sparse, noise, type, algorithm)

def add_datapass_indicator(max_iter, m, num_samp):
    for i in range(1, (num_samp*max_iter)/m+1):
        plt.axvline(i * (m/num_samp), linestyle="-.", color="0.75")
    
def save_plot(sparse, noise, type, algorithm):
    plt.title(get_title(sparse, noise, type, algorithm))
    plt.savefig(get_path(sparse, noise, type, algorithm))
    
def meta_plot(max_iter, results, sparse, noise, algorithm, m, num_samp, lmbda=None):
    plot_residual(max_iter, results.get_residuals(), sparse, noise, m, num_samp)
    plot_onenorm(max_iter, results.get_onenorm(), sparse, noise, m, num_samp)
    plot_moder(max_iter, results.get_moder(), sparse, noise, m, num_samp)
    plot_residual_vs_sparsity(results.get_residuals(), results.get_onenorm(), sparse, noise, m, num_samp)
    
    if (lmbda != None):
        plot_z_nonzeros(max_iter, results.get_z_history_nonzeros(), lmbda, sparse, noise)
    else:
        plot_x_nonzeros(max_iter, results.get_x_history_nonzeros(), sparse, noise)
        
    
