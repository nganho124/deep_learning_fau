import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        """
        Initialize Dropout layer.
        
        Args:
            probability: fraction of units to KEEP (not drop)
                        e.g., 0.8 means keep 80%, drop 20%
        """
        super().__init__()
        self.trainable = False  # No learnable parameters
        self.probability = probability
        self.mask = None  # Store mask for backward pass
    
    def forward(self, input_tensor):
        """
        Forward pass with inverted dropout.
        
        Training: randomly zero out neurons and scale
        Testing: pass through unchanged
        """
        if self.testing_phase:
            # Testing: just pass through
            return input_tensor
        else:
            # Training: apply dropout
            # Create random mask: 1 with probability p, 0 with probability (1-p)
            self.mask = np.random.random(input_tensor.shape) < self.probability
            
            # Apply mask and scale by 1/probability (inverted dropout)
            return (input_tensor * self.mask) / self.probability
    
    def backward(self, error_tensor):
        """
        Backward pass: gradient flows only through kept neurons.
        
        The same mask and scaling is applied to the gradient.
        """
        # Gradient only flows through neurons that were kept
        # Same scaling as forward pass
        return (error_tensor * self.mask) / self.probability