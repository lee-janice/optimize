def main():
    # ------ CONFIGURE PARAMETERS ------
    params = set_params.Params()
    m = params.m         
    n = params.n         
    num_samp = params.num_samp    
    max_iter = params.max_iter
    
    lmbda = params.lmbda 
    eta = params.eta
    epsilon = params.epsilon
    
    sparse = params.sparse 
    noise = params.noise
    flipping = params.flipping
    # ------ EXECUTE ------
    
