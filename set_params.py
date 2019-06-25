class Params:
    def __init__(self):
        self.m = 100000
        self.n = 2000
        self.num_samp = 300 
        self.max_iter = 300 
        
        self.lmbda = 2.0
        self.eta = 0.5
        self.epsilon = 1e-6
        self.beta_1 = .9
        self.beta_2 = .999

        self.sparse = False 
        self.noise = True 
        self.flipping = True
