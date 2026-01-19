import numpy as np

class Sgd:

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):

        update_weights = weight_tensor - self.learning_rate * gradient_tensor

        return update_weights
    
class SgdWithMomentum:
    
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor

        update_weights = weight_tensor + self.velocity

        return update_weights
    
class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8

        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        
        # Update biased second moment estimate
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        
        # Update weights
        updated_weights = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_weights