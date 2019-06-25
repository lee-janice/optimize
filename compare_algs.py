"""
@authors: Jimmy Singh and Janice Lee 
@date: June 24th, 2019
"""
from lb_classic import *
from lb_modified import *
from adagrad import *
from adagrad_lb_classic import *
from adagrad_lb_modified import *
from plot import *
import matplotlib.pyplot as plt

def main(): 
    # ------ CONFIGURE PARAMETERS ------
    m = 100000         # rows of A 
    n = 2000          # columns of A (rows of x_true and y_true)
    num_samp = 300    # rows of A and y to sample, num_samp < n
    max_iter = 300
    sparse = False
    noise = True
    lb_lmbda = 3.0
    adagrad_lmbda = 3.0
    eta = 0.5
    epsilon = 1e-6
    
    # ------ EXECUTE ------
    lb_classic_results = lb_classic(m, n, num_samp, max_iter, lb_lmbda, sparse, noise)
    lb_modified_results = lb_modified(m, n, num_samp, max_iter, lb_lmbda, sparse, noise, flipping=False)
    lb_modified_flipping_results = lb_modified(m, n, num_samp, max_iter, lb_lmbda, sparse, noise, flipping=True)
    adagrad_results = adagrad(m, n, num_samp, max_iter, sparse, noise, eta, epsilon)
    adagrad_lb_classic_results = adagrad_lb_classic(m, n, num_samp, max_iter, adagrad_lmbda, sparse, noise, eta, epsilon)
    adagrad_lb_modified_results = adagrad_lb_modified(m, n, num_samp, max_iter, adagrad_lmbda, sparse, noise, eta, epsilon, flipping=False)
    adagrad_lb_modified_flipping_results = adagrad_lb_modified(m, n, num_samp, max_iter, adagrad_lmbda, sparse, noise, eta, epsilon, flipping=True)
    
    # ------ LB ------ 
    lb_legend = ["Classic", "Modified", "Modified with flipping"]
    
    lb_residuals = np.zeros((max_iter, 3))
    lb_residuals[:, 0] = lb_classic_results.get_residuals()
    lb_residuals[:, 1] = lb_modified_results.get_residuals()
    lb_residuals[:, 2] = lb_modified_flipping_results.get_residuals()
    plot_residual(max_iter, lb_residuals, sparse, noise)
    plt.legend(lb_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "residual", "COMPARE-lb")
    
    lb_onenorm = np.zeros((max_iter, 3))
    lb_onenorm[:, 0] = lb_classic_results.get_onenorm()
    lb_onenorm[:, 1] = lb_modified_results.get_onenorm()
    lb_onenorm[:, 2] = lb_modified_flipping_results.get_onenorm()
    plot_onenorm(max_iter, lb_onenorm, sparse, noise)
    plt.legend(lb_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "1-norm", "COMPARE-lb")
    
    lb_moder = np.zeros((max_iter, 3))
    lb_moder[:, 0] = lb_classic_results.get_moder()
    lb_moder[:, 1] = lb_modified_results.get_moder()
    lb_moder[:, 2] = lb_modified_flipping_results.get_moder()
    plot_moder(max_iter, lb_moder, sparse, noise)
    plt.legend(lb_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "model-error", "COMPARE-lb")
    
    plot_residual_vs_sparsity(lb_residuals, lb_onenorm, sparse, noise)
    plt.legend(lb_legend)
    save_plot(sparse, noise, "residual-vs-sparsity", "COMPARE-lb")
    
    # ------ ADAGRAD ------ 
    adagrad_legend = ["+LB (Classic)", "+LB (Modified)", "+LB (Modified with flipping)"]
    
    adagrad_residuals = np.zeros((max_iter, 3))
    adagrad_residuals[:, 0] = adagrad_lb_classic_results.get_residuals()
    adagrad_residuals[:, 1] = adagrad_lb_modified_results.get_residuals()
    adagrad_residuals[:, 2] = adagrad_lb_modified_flipping_results.get_residuals()
    # adagrad_residuals[:, 3] = adagrad_results.get_residuals()
    plot_residual(max_iter, adagrad_residuals, sparse, noise)
    plt.legend(adagrad_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "residual", "COMPARE-adagrad")
    
    adagrad_onenorm = np.zeros((max_iter, 3))
    adagrad_onenorm[:, 0] = adagrad_lb_classic_results.get_onenorm()
    adagrad_onenorm[:, 1] = adagrad_lb_modified_results.get_onenorm()
    adagrad_onenorm[:, 2] = adagrad_lb_modified_flipping_results.get_onenorm()
    # adagrad_onenorm[:, 3] = adagrad_results.get_onenorm()
    plot_onenorm(max_iter, adagrad_onenorm, sparse, noise)
    plt.legend(adagrad_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "1-norm", "COMPARE-adagrad")
    
    adagrad_moder = np.zeros((max_iter, 3))
    adagrad_moder[:, 0] = adagrad_lb_classic_results.get_moder()
    adagrad_moder[:, 1] = adagrad_lb_modified_results.get_moder()
    adagrad_moder[:, 2] = adagrad_lb_modified_flipping_results.get_moder()
    # adagrad_moder[:, 3] = adagrad_results.get_moder()
    plot_moder(max_iter, adagrad_moder, sparse, noise)
    plt.legend(adagrad_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "model-error", "COMPARE-adagrad")
    
    plot_residual_vs_sparsity(adagrad_residuals, adagrad_onenorm, sparse, noise)
    plt.legend(adagrad_legend)
    save_plot(sparse, noise, "residual-vs-sparsity", "COMPARE-adagrad")
    
    # ------ LB VS ADAGRAD ------ 
    lb_vs_adagrad_legend = ["LB", "MLB", "MLB-F", "ADAGRAD-LB", "ADAGRAD-MLB", "ADAGRAD-MLB-F"]
    lb_vs_adagrad_residuals = np.zeros((max_iter, 6))
    lb_vs_adagrad_residuals[:, :3] = lb_residuals
    lb_vs_adagrad_residuals[:, 3:7] = adagrad_residuals
    plot_residual(max_iter, lb_vs_adagrad_residuals, sparse, noise)
    plt.legend(lb_vs_adagrad_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "residual", "COMPARE-lb-vs-adagrad")
        
    lb_vs_adagrad_onenorm = np.zeros((max_iter, 6))
    lb_vs_adagrad_onenorm[:, :3] = lb_onenorm
    lb_vs_adagrad_onenorm[:, 3:7] = adagrad_onenorm
    plot_onenorm(max_iter, lb_vs_adagrad_onenorm, sparse, noise)
    plt.legend(lb_vs_adagrad_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "1-norm", "COMPARE-lb-vs-adagrad")
    
    lb_vs_adagrad_moder = np.zeros((max_iter, 6))
    lb_vs_adagrad_moder[:, :3] = lb_moder
    lb_vs_adagrad_moder[:, 3:7] = adagrad_moder
    plot_moder(max_iter, lb_vs_adagrad_moder, sparse, noise)
    plt.legend(lb_vs_adagrad_legend)
    add_datapass_indicator(max_iter, m, num_samp)
    save_plot(sparse, noise, "model-error", "COMPARE-lb-vs-adagrad")
    
    plot_residual_vs_sparsity(lb_vs_adagrad_residuals, lb_vs_adagrad_onenorm, sparse, noise)
    plt.legend(lb_vs_adagrad_legend)
    save_plot(sparse, noise, "residual-vs-sparsity", "COMPARE-lb-vs-adagrad")
    
if __name__ == '__main__':
    main()
