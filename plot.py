"""
Plots the results of the methods and saves them as figures
@authors: Jimmy Singh and Janice Lee
@date: June 11th, 2019
"""
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
plt.style.use('ggplot')
import os 

class Plot:
    def __init__(self, params):
        self.max_iter = params.max_iter
        self.m = params.m 
        self.num_samp = params.num_samp
        
        self.sparse = params.sparse 
        self.noise = params.noise 
        self.lmbda = params.lmbda
        
    def get_title(self, type):
        """
        Generates figure/plot titles
        params: 
            type (str): the type of data to be plotted (residual, 1-norm,  model error)
        returns:
            the generated figure/plot title 
        """
        title = self.algorithm.upper() + " ("
        # get sparse title 
        if (self.sparse): 
            title += "sparse soln, "
        else:
            title += "exponentally decaying soln, "
        # get noise title 
        if (self.noise):
            title += "w/noise) - "
        else: 
            title += "w/out noise) - "
        title += type.upper()
        return title 

    def get_path(self, type):
        """
        Generates the path to save the figure/plots 
        params: 
            type (str): the type of data to be plotted (residual, 1-norm,  model error)
        returns:
            the generated figure/plot filename 
        """
        if (self.sparse): 
            subclass = "sparse_"
        else:
            subclass = "randExpDecay_"
        if (self.noise):
            subclass += "noise/"
        else: 
            subclass += "noNoise/"
            
        if (not os.path.exists("plots/")):
            os.mkdir("plots/")
        
        if (not os.path.exists("plots/" + self.algorithm + "/")):
            os.mkdir("plots/" + self.algorithm + "/")
        
        if (not os.path.exists("plots/" + self.algorithm + "/" + subclass)):
            os.mkdir("plots/" + self.algorithm + "/" + subclass)
            
        filename = "plots/" + self.algorithm + "/" + subclass + type
        # get sparse filename 
        if (self.sparse): 
            filename += "_sparse_"
        else:
            filename += "_randExpDecay_"
        # get noise filename 
        if (self.noise):
            filename += "noise.png"
        else: 
            filename += "noNoise.png"
        return filename
        
    def plot_residual(self): 
        """
        Plots number of iterations vs. residual and saves the plot
        params: none
        returns: none 
        """
        plt.clf()
        plt.plot(range(1, self.max_iter+1), self.residual)
        
        plt.xlabel("Number of iterations")
        plt.ylabel("Residual " + r"$||Ax-b||_{2}$")
        
        self.add_datapass_indicator()
        self.save_plot("residual")

    def plot_moder(self): 
        """
        Plots number of iterations vs. model error and saves the plot
        params: none
        returns: none 
        """
        plt.clf()
        plt.plot(range(1, self.max_iter+1), self.moder)
        
        plt.xlabel("Number of iterations")
        plt.ylabel("Model error " + r"$\frac{||x^*-x||_{2}}{||x^*||_{2}}$")
        
        self.add_datapass_indicator()
        self.save_plot("model-error")
        
    def plot_onenorm(self): 
        """
        Plots number of iterations vs. 1-norm and saves the plot
        params: none
        returns: none 
        """
        plt.clf()
        plt.plot(range(1, self.max_iter+1), self.onenorm)
        
        plt.xlabel("Number of iterations")
        plt.ylabel("1-norm " + r"$||x||_{1}$")
        
        self.add_datapass_indicator()
        self.save_plot("1-norm")
        
    def plot_residual_vs_sparsity(self): 
        """
        Plots 1-norm vs. residual and saves the plot
        params: none
        returns: none 
        """
        plt.clf()
        plt.plot(self.onenorm, self.residual)
        
        plt.xlabel("1-norm " + r"$||x||_{1}$")
        plt.ylabel("Residual " + r"$||Ax-b||_{2}$")
        
        self.save_plot("residual-vs-sparsity")
        
    def plot_x_nonzeros(self):
        """
        Plots progression of z_k values and saves the plot 
        params: none
        returns: none 
        """
        plt.clf()
        plt.plot(range(1, self.max_iter+1), self.x_k)
        
        plt.xlabel("Number of iterations")
        plt.ylabel(r"$x_k$")
        
        self.add_datapass_indicator()
        self.save_plot("x-nonzeros")

    def plot_z_nonzeros(self):
        """
        Plots progression of z_k values and saves the plot 
        params:
            none
        returns: 
            none 
        """
        plt.clf()
        plt.plot(range(1, self.max_iter+1), self.z_k)
        
        plt.axhline(y=self.lmbda, linestyle="--")
        plt.axhline(y=-self.lmbda, linestyle="--")
        
        plt.xlabel("Number of iterations")
        plt.ylabel(r"$z_k$")
        
        self.add_datapass_indicator()
        self.save_plot("z-nonzeros")
        
    def plot_t_nonzeros(self):
        plt.clf()
        plt.plot(range(1, self.max_iter+1), self.t_small, linestyle=":", label="Small value")
        plt.plot(range(1, self.max_iter+1), self.t_large, linestyle=":", label="Large value")
        
        plt.legend()
        plt.xlabel("Number of iterations")
        plt.ylabel(r"$t_k$")
        
        self.add_datapass_indicator()
        self.save_plot("t-nonzeros")

    def add_datapass_indicator(self):
        for i in range(1, (int((self.num_samp*self.max_iter)/self.m+1))):
            plt.axvline(i * (self.m/self.num_samp), linestyle="-.", color="0.75")
        
    def save_plot(self, type):
        if (self.legend != None):
            plt.legend(self.legend)
        plt.title(self.get_title(type))
        plt.savefig(self.get_path(type))
        
    def plot_all(self):
        self.plot_residual()
        self.plot_onenorm()
        self.plot_moder()
        self.plot_residual_vs_sparsity()
        self.plot_t_nonzeros()
        
        if (self.thresholding):
            self.plot_z_nonzeros()
        else:
            self.plot_x_nonzeros()
            
    def update_results(self, results):
        self.residual = results.get_residuals()
        self.onenorm = results.get_onenorm()
        self.moder = results.get_moder()
        self.x_k = results.get_x_history_nonzeros()
        self.z_k = results.get_z_history_nonzeros()
        self.t_small = results.get_t_history_small()
        self.t_large = results.get_t_history_large()
        
    def update_algorithm(self, algorithm, results, thresholding, legend=None):
        self.residual = results.get_residuals()
        self.onenorm = results.get_onenorm()
        self.moder = results.get_moder()
        self.x_k = results.get_x_history_nonzeros()
        self.z_k = results.get_z_history_nonzeros()
        self.t_k = results.get_t_history()
        
        self.t_small = results.get_t_history_small()
        self.t_large = results.get_t_history_large()
        
        self.algorithm = algorithm 
        self.thresholding = thresholding 
        self.legend = legend
            
        
