from lb_classic import *  
from lb_modified import *  
from adagrad import *  
from adagrad_lb_classic import *  
from adagrad_lb_modified import *  
from adam import *  
from adam_lb_classic import * 
from adam_lb_modified import *
import plot 
import numpy as np 
np.random.seed(12)

def main():
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    
    lbc = lbm = lbm_wf = adag = adag_lbc = adag_lbm = adag_lbm_wf = adm = adam_lbc = adam_lbm = adam_lbm_wf = False
    
    lbc = True 
    lbm = True 
    lbm_wf = True
    
    # adag = True 
    # adag_lbc = True 
    # adag_lbm = True 
    # adag_lbm_wf = True 
    
    # adm = True 
    adam_lbc = True 
    adam_lbm = True 
    adam_lbm_wf = True
    
    # ------ EXECUTE AND PLOT------
    plt = plot.Plot(params)
    algs = {}
    if (lbc):
        alg = "lb-classic"
        algs[alg] = lb_classic(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
        
    if (lbm):
        alg = "lb-modified"
        params.flipping = False
        algs[alg] = lb_modified(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
    
    if (lbm_wf):
        alg = "lb-modified-w-flipping"
        params.flipping = True
        algs[alg] = adam_lb_modified(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
        
    if (adag):
        alg = "adagrad"
        algs[alg] = adagrad(params)
        plt.update_algorithm(alg, algs[alg], thresholding=False)
        plt.plot_all()
        
    if (adag_lbc):
        alg = "adagrad-lb-classic"
        algs[alg] = adagrad_lb_classic(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
        
    if (adag_lbm):
        alg = "adagrad-lb-modified"
        params.flipping = False 
        algs[alg] = adagrad_lb_modified(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
    
    if (adag_lbm_wf):
        alg = "adagrad-lb-modified-w-flipping"
        params.flipping = True 
        algs[alg] = adagrad_lb_modified(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
        
    if (adm):
        alg = "adam"
        algs[alg] = adam(params)
        plt.update_algorithm(alg, algs[alg], thresholding=False)
        plt.plot_all()
        
    if (adam_lbc):
        alg = "adam-lb-classic"
        algs[alg] = adam_lb_classic(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
        
    if (adam_lbm):
        alg = "adam-lb-modified"
        params.flipping = False
        algs[alg] = adam_lb_modified(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
        
    if (adam_lbm_wf):
        alg = "adam-lb-modified-w-flipping"
        params.flipping = True
        algs[alg] = adam_lb_modified(params)
        plt.update_algorithm(alg, algs[alg], thresholding=True)
        plt.plot_all()
    
    for alg in sorted(algs.keys()):
        print(alg + ": " + str(algs[alg].get_percent_nonzeros_recovered()))
        
    """
    ----------------------------------------
    SPARSE, NO NOISE 
    ----------------------------------------
    adagrad-lb-classic: 0.87
    adagrad-lb-modified: 0.7125
    adagrad-lb-modified-w-flipping: 0.8
    
    adam-lb-classic: 0.87
    adam-lb-modified: 0.93
    adam-lb-modified-w-flipping: 0.9575
    
    lb-classic: 0.8675
    lb-modified: 0.7075
    lb-modified-w-flipping: 0.94
    ----------------------------------------
    SPARSE, NOISE 
    ----------------------------------------
    adagrad-lb-classic: 0.8725
    adagrad-lb-modified: 0.7
    adagrad-lb-modified-w-flipping: 0.785
    
    adam-lb-classic: 0.6875
    adam-lb-modified: 0.925
    adam-lb-modified-w-flipping: 0.9125
    
    lb-classic: 0.8625
    lb-modified: 0.7225
    lb-modified-w-flipping: 0.9275
    ----------------------------------------
    EXP DECAY, NO NOISE 
    ----------------------------------------
    adagrad-lb-classic: 0.332
    adagrad-lb-modified: 0.1855
    adagrad-lb-modified-w-flipping: 0.285
    
    adam-lb-classic: 0.309
    adam-lb-modified: 0.3545
    adam-lb-modified-w-flipping: 0.433
    
    lb-classic: 0.297
    lb-modified: 0.1945
    lb-modified-w-flipping: 0.412
    ----------------------------------------
    EXP DECAY, NOISE 
    ----------------------------------------
    adagrad-lb-classic: 0.3165
    adagrad-lb-modified: 0.1875
    adagrad-lb-modified-w-flipping: 0.2835
    
    adam-lb-classic: 0.354
    adam-lb-modified: 0.3815
    adam-lb-modified-w-flipping: 0.413
    
    lb-classic: 0.3165
    lb-modified: 0.1965
    lb-modified-w-flipping: 0.3975
    ----------------------------------------
    """
    
if __name__ == '__main__':
    main()
         
