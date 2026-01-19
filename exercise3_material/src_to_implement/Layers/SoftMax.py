from Layers.Base import *
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Numerical stability: subtract max from each row
        shifted = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        
        # Compute exponentials
        exp_values = np.exp(shifted)
        
        # Normalize by sum of each row
        self.output_tensor = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Store for backward pass (the hint!)
        return self.output_tensor
    
    def backward(self, error_tensor):
        # Compute sum of (error * output) for each sample
        weighted_sum = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        
        # Compute error for previous layer
        error_prev = self.output_tensor * (error_tensor - weighted_sum)
        
        return error_prev