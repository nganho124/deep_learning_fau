import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activation = None  # Store activation for backward pass
    
    def forward(self, input_tensor):
        """
        Forward pass: σ(x) = 1 / (1 + e^(-x))
        """
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation
    
    def backward(self, error_tensor):
        """
        Backward pass: gradient = σ * (1 - σ) * error
        
        Uses stored activation (not input) for efficiency.
        """
        # Gradient of sigmoid: σ(x) * (1 - σ(x))
        gradient = self.activation * (1 - self.activation)
        return gradient * error_tensor