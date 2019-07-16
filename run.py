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
import matplotlib.pyplot as mat
mat.style.use('seaborn-poster')
mat.style.use('ggplot')
np.random.seed(0)

def run(params, plt, lbc, lbm, lbm_wf, adag, adag_lbc, adag_lbm, adag_lbm_wf, adm, adam_lbc, adam_lbm, adam_lbm_wf):    
    # ------ EXECUTE AND PLOT------
    algs = {}
    if (lbc):
        alg = "lb-classic"
        algs[alg] = lb_classic(params)
        # plt.update_algorithm(alg, algs[alg], thresholding=True)
        # plt.plot_all()
        
    if (lbm):
        alg = "lb-modified"
        params.flipping = False
        algs[alg] = lb_modified(params)
        # plt.update_algorithm(alg, algs[alg], thresholding=True)
        # plt.plot_all()
    
    if (lbm_wf):
        alg = "lb-modified-w-flipping"
        params.flipping = True
        algs[alg] = adam_lb_modified(params)
        # plt.update_algorithm(alg, algs[alg], thresholding=True)
        # plt.plot_all()
        
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
        # plt.update_algorithm(alg, algs[alg], thresholding=False)
        # plt.plot_all()
        
    if (adam_lbc):
        alg = "adam-lb-classic"
        algs[alg] = adam_lb_classic(params)
        # plt.update_algorithm(alg, algs[alg], thresholding=True)
        # plt.plot_all()
        
    if (adam_lbm):
        alg = "adam-lb-modified"
        params.flipping = False
        algs[alg] = adam_lb_modified(params)
        # plt.update_algorithm(alg, algs[alg], thresholding=True)
        # plt.plot_all()
        
    if (adam_lbm_wf):
        alg = "adam-lb-modified-w-flipping"
        params.flipping = True
        algs[alg] = adam_lb_modified(params)
        # plt.update_algorithm(alg, algs[alg], thresholding=True)
        # plt.plot_all()
    
    return algs

def main():
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    plt = plot.Plot(params)
    # ------ FLAGGING ALGS TO RUN ------
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
    #-----------------------------------
    algs = run(params, plt, lbc, lbm, lbm_wf, adag, adag_lbc, adag_lbm, adag_lbm_wf, adm, adam_lbc, adam_lbm, adam_lbm_wf)
    # dicts = []
    # for i in range(1, 5):
    #     params.lmbda = float(i)
    #     algs = run(params, plt, lbc, lbm, lbm_wf, adag, adag_lbc, adag_lbm, adag_lbm_wf, adm, adam_lbc, adam_lbm, adam_lbm_wf)
    #     dicts.append(algs)
    # 
    # for dict in dicts:
    #     for alg in sorted(dict.keys()):
    #         print(alg + ": " + str(dict[alg].get_percent_nonzeros_recovered()))
    #         print("\tmodel error - " + str(dict[alg].get_moder()[params.max_iter-1]))
    #     print("")
    
    for name in algs.keys():
        print(name)
    
    legend = ["CLB", "MLB", "MLB-F", "ADAM-CLB", "ADAM-MLB", "ADAM-MLB-F"]
    
    mat.clf()
    for alg in algs.values(): 
        mat.plot(range(1, 301), alg.moder)
    mat.xlabel("Number of iterations", size=20)
    mat.ylabel("Model error " + r"$\frac{||x^*-x||_{2}}{||x^*||_{2}}$", size=20)
    
    plt.add_datapass_indicator()
    mat.legend(legend)
    mat.savefig("plots/_compare/moder.png")

    mat.clf()
    for alg in algs.values(): 
        mat.plot(range(1, 301), alg.residuals)
    mat.xlabel("Number of iterations", size=20)
    mat.ylabel("Residual " + r"$\frac{||Ax-b||_{2}}{||b||_2}$", size=20)
    
    plt.add_datapass_indicator()
    mat.legend(legend, loc="upper right")
    mat.savefig("plots/_compare/residual.png")
    
    mat.clf()
    for alg in algs.values(): 
        mat.plot(alg.onenorm, alg.residuals)
    mat.xlabel(r"$L1$ norm " + r"$||x||_{1}$", size=20)
    mat.ylabel("Residual " + r"$\frac{||Ax-b||_{2}}{||b||_2}$", size=20)
    
    mat.legend(legend)
    mat.savefig("plots/_compare/res-vs-sparsity.png")
    
    i = 0
    for alg in algs.values(): 
        mat.clf()
        mat.plot(range(1, 301), alg.get_z_history_nonzeros())
        
        mat.xlabel("Number of iterations", size=20)
        mat.ylabel(r"$z_k$", size=20)
        
        mat.axhline(y=params.lmbda, linestyle="--")
        mat.axhline(y=-params.lmbda, linestyle="--")
        
        mat.legend([legend[i]], handlelength=0)

        plt.add_datapass_indicator()
        mat.savefig("plots/_compare/z-nonzeros"+str(i)+".png")
        
        i += 1
    
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
    
    
    
    
    
    
    adam-lb-classic: 0.5495
        model error - 0.7916572919866612
    adam-lb-modified: 0.747
            model error - 0.43828161749793015
    adam-lb-modified-w-flipping: 0.7485
            model error - 0.4610652392602479
    lb-classic: 0.539
            model error - 0.06994936847970029
    lb-modified: 0.407
            model error - 0.11171179913059176
    lb-modified-w-flipping: 0.725
            model error - 0.4327165986283041
            
            
            
    adam-lb-classic: 0.3765
            model error - 0.41341109078781985
    adam-lb-modified: 0.594
            model error - 0.22180039401252127
    adam-lb-modified-w-flipping: 0.5665
            model error - 0.258803628845938
            
    lb-classic: 0.349
            model error - 0.05027406816282646
    lb-modified: 0.2395
            model error - 0.09374244104691382
    lb-modified-w-flipping: 0.5525
            model error - 0.22843130255130975
            
            
            
    adam-lb-classic: 0.3425
            model error - 0.39120864745785405
    adam-lb-modified: 0.4545
            model error - 0.08630201610245969
    adam-lb-modified-w-flipping: 0.4545
            model error - 0.1656206493144285
            
    lb-classic: 0.271
            model error - 0.056471671814865856
    lb-modified: 0.179
            model error - 0.10262477615277157
    lb-modified-w-flipping: 0.4825
            model error - 0.15406531602143575
            
            
            
    adam-lb-classic: 0.2965
            model error - 0.32817767031652767
    adam-lb-modified: 0.379
            model error - 0.06611923777482336
    adam-lb-modified-w-flipping: 0.379
            model error - 0.11444434817733892
            
    lb-classic: 0.2435
            model error - 0.06087186299595924
    lb-modified: 0.1745
            model error - 0.12202274097433019
    lb-modified-w-flipping: 0.4135
            model error - 0.12403922983369688
    """
    
if __name__ == '__main__':
    main()
         
