import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activation = None  # Store activation for backward pass
    
    def forward(self, input_tensor):
        """
        Forward pass: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Using np.tanh for numerical stability.
        """
        self.activation = np.tanh(input_tensor)
        return self.activation
    
    def backward(self, error_tensor):
        """
        Backward pass: gradient = (1 - tanh²(x)) * error
        
        Uses stored activation (not input) for efficiency.
        """
        # Gradient of tanh: 1 - tanh²(x)
        gradient = 1 - self.activation ** 2
        return gradient * error_tensor