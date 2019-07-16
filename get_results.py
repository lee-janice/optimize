"""
Class to store and update the results of optimization algorithms 
@authors: Jimmy Singh and Janice Lee 
@date: June 24th, 2019
"""
import numpy as np
import numpy.linalg as la

class Results: 
    def __init__(self, max_iter, n, x_true):
        self.residuals = np.zeros((max_iter))
        self.onenorm = np.zeros((max_iter))
        self.moder = np.zeros((max_iter))
        self.x_history = np.zeros((max_iter, n))
        self.z_history = np.zeros((max_iter, n))
        self.t_history = np.zeros((max_iter, n))
        self.x_true = x_true
        self.idx_nonzeros = np.argwhere(x_true!=0)[:, 0]
        self.i = 0
    
    def update(self, residual, b_sub, n, x_k, z_k, t_k, adaptive):
        self.update_iteration()
        self.update_residuals(residual, b_sub)
        self.update_onenorm(x_k)
        self.update_moder(x_k)
        self.update_x_history(x_k, n)
        self.update_z_history(z_k, n)
        self.update_t_history(t_k, n, adaptive)
        
        
    def update_iteration(self):
        self.i += 1
        
    def update_residuals(self, residual, b_sub):
        self.residuals[self.i-1] = la.norm(residual, 2) / la.norm(b_sub, 2)
        
    def update_onenorm(self, x_k):
        self.onenorm[self.i-1] = la.norm(x_k, 1)
    
    def update_moder(self, x_k):
        self.moder[self.i-1] = la.norm(self.x_true - x_k, 2) / la.norm(self.x_true, 2)

    def update_x_history(self, x_k, n):
        self.x_history[self.i-1, :] = x_k.reshape(n,)
    
    def update_z_history(self, z_k, n):
        self.z_history[self.i-1, :] = z_k.reshape(n,)
    
    def update_t_history(self, t_k, n, adaptive):
        if adaptive:
            self.t_history[self.i-1, :] = t_k.reshape(n,)
        else:
            self.t_history[self.i-1, :] = t_k
            
    def get_t_history_small(self):
        return self.t_history[:,self.idx_nonzeros][:,14]
    
    def get_t_history_large(self):
        return self.t_history[:,self.idx_nonzeros][:,8]
        
    def get_residuals(self):
        return self.residuals 
        
    def get_onenorm(self):
        return self.onenorm 
        
    def get_moder(self):
        return self.moder  
        
    def get_x_history(self):
        return self.x_history 
        
    def get_z_history(self):
        return self.z_history
        
    def get_t_history(self):
        return self.t_history
    
    def get_x_history_nonzeros(self):
        return self.x_history[:, self.idx_nonzeros[:25]]
        
    def get_z_history_nonzeros(self):
        return self.z_history[:, self.idx_nonzeros[:25]]
        
    def get_percent_nonzeros_recovered(self):
        return float(len(np.argwhere(self.x_history[-1, self.idx_nonzeros]!=0)[:,0])) / len(self.idx_nonzeros)
