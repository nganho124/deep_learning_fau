import numpy as np

class Optimizer:

    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:

            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum(Optimizer):
    
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.m = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        
        # First shrink weights if regularizer is set
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        
        # v_k = mu * v_{k-1} - lr * gradient
        self.m = self.momentum_rate * self.m - self.learning_rate * gradient_tensor
        
        # w_{k+1} = shrinked_weights + v_k
        return weight_tensor + self.m
    
class Adam(Optimizer):

    def __init__(self, learning_rate: float, mu: float, rho: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu      # beta1 - for first moment
        self.rho = rho    # beta2 - for second moment
        self.v = 0        # first moment estimate
        self.r = 0        # second moment estimate
        self.k = 0        # iteration counter
        self.eps = 1e-8   # small constant for numerical stability
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # First shrink weights if regularizer is set
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        
        # Increment iteration counter
        self.k += 1
        
        # Update biased first moment estimate
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        
        # Update biased second moment estimate
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)
        
        # Compute bias-corrected first moment estimate
        v_hat = self.v / (1 - self.mu ** self.k)
        
        # Compute bias-corrected second moment estimate
        r_hat = self.r / (1 - self.rho ** self.k)
        
        # Update weights (using shrinked weights)
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)