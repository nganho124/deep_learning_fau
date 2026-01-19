import numpy as np


class L2_Regularizer:
    """
    L2 Regularization (Ridge)
    - Penalizes squared magnitude of weights
    - Encourages small weights
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        """
        Gradient of L2 regularization term: d/dw (α * w²) = 2αw
        But commonly written as just αw (factor of 2 absorbed into α)
        """
        return self.alpha * weights
    
    def norm(self, weights):
        """
        L2 norm contribution to loss: α * Σ(w²)
        """
        return self.alpha * np.sum(weights ** 2)


class L1_Regularizer:
    """
    L1 Regularization (Lasso)
    - Penalizes absolute value of weights
    - Encourages sparsity (weights become exactly 0)
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        """
        Subgradient of L1 regularization term: d/dw (α * |w|) = α * sign(w)
        """
        return self.alpha * np.sign(weights)
    
    def norm(self, weights):
        """
        L1 norm contribution to loss: α * Σ|w|
        """
        return self.alpha * np.sum(np.abs(weights))