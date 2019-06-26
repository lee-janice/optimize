from lb_classic import *  
from lb_modified import *  
from adagrad import *  
from adagrad_lb_classic import *  
from adagrad_lb_modified import *  
from adam import *  
from adam_lb_classic import * 
import plot 

def main():
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    
    lbc = lbm = adag = adag_lbc = adag_lbm = adm = adam_lbc = adam_lbm = False
    lbc = True 
    lbm = True 
    adag = True 
    adag_lbc = True 
    adag_lbm = True 
    adm = True 
    adam_lbc = True 
    adam_lbm = True 
    
    # ------ EXECUTE AND PLOT------
    plt = plot.Plot(params)
    if (lbc):
        lb_classic_results = lb_classic(params)
        plt.update_algorithm("lb-classic", lb_classic_results, thresholding=True)
        plt.plot_all()
        
    if (lbm):
        lb_modified_results = lb_modified(params)
        if (params.flipping):
            plt.update_algorithm("lb-modified-w-flipping", lb_modified_results, thresholding=True)
        else: 
            plt.update_algorithm("lb-modified", lb_modified_results, thresholding=True)
        plt.plot_all()
        
    if (adag):
        adagrad_results = adagrad(params)
        plt.update_algorithm("adagrad", adagrad_results, thresholding=False)
        plt.plot_all()
        
    if (adag_lbc):
        adagrad_lb_classic_results = adagrad_lb_classic(params)
        plt.update_algorithm("adagrad-lb-classic", adagrad_lb_classic_results, thresholding=True)
        plt.plot_all()
        
    if (adag_lbm):
        adagrad_lb_modified_results = adagrad_lb_modified(params)
        if (params.flipping):
            plt.update_algorithm("adagrad-lb-modified-w-flipping", lb_modified_results, thresholding=True)
        else: 
            plt.update_algorithm("adagrad-lb-modified", lb_modified_results, thresholding=True)
        plt.plot_all()
        
    if (adm):
        adam_results = adam(params)
        plt.update_algorithm("adam", adam_results, thresholding=False)
        plt.plot_all()
        
    if (adam_lbc):
        adam_lb_classic_results = adam_lb_classic(params)
        plt.update_algorithm("adam-lb-classic", adam_lb_classic_results, thresholding=True)
        plt.plot_all()
        
    # if (adam_lbm):
    
if __name__ == '__main__':
    main()
         
