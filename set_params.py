class Params:
    def __init__(self):
        """
        Initializes the following values:
            m (int): rows of A 
            n (int): columns of A / rows of x and b
            num_samp (int): rows of A and b to sample, num_samp < n 
            max_iter (int): number of iterations to run 
            
            lmbda (float): the thresholding parameter 
            eta (float): the desired learning rate 
            epsilon (float): a small constant to avoid division by zero in calculation of step size 
            beta_1 (float): used for exponentially moving average of the gradient mean in ADAM
            beta_2 (float): used for exponentially moving average of the gradient variance in ADAM
            
            sparse (bool): true if the soln is sparse 
            noise (bool): true if the data contains noise 
            flipping (bool): true if step sizes should only be updated when values cross threshold
        """
        self.m = 20000
        self.n = 2000
        self.num_samp = 200 
        self.max_iter = 300 
        
        self.lmbda = 3.0
        self.eta = 0.5
        self.epsilon = 1e-6
        self.beta_1 = .9
        self.beta_2 = .999

        self.sparse = True   
        self.noise = False
        self.flipping = True
