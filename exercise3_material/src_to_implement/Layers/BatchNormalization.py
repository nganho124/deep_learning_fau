import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        """
        Initialize Batch Normalization layer.
        
        Args:
            channels: number of channels (features in vector case, 
                     or channels in image case)
        """
        super().__init__()
        self.trainable = True
        self.channels = channels
        
        # Initialize weights (gamma) and bias (beta)
        self.initialize(None, None)
        
        # Running statistics for testing phase (moving average)
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9  # For moving average
        
        # Optimizer (separate for weights and bias)
        self._optimizer = None
        self._bias_optimizer = None
        
        # Store for backward pass
        self.input_tensor = None
        self.input_normalized = None
        self.mean = None
        self.var = None
        self.eps = 1e-10
        
        # Gradient storage
        self._gradient_weights = None
        self._gradient_bias = None
        
        # Track input shape for reformat
        self.is_convolutional = False
        self.original_shape = None
    
    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize gamma (weights) to ones and beta (bias) to zeros.
        Ignores the provided initializers.
        """
        self.weights = np.ones(self.channels)  # gamma
        self.bias = np.zeros(self.channels)    # beta
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self._bias_optimizer = copy.deepcopy(value)
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    def reformat(self, tensor):
        """
        Reformat tensor between image-like (4D) and vector-like (2D).
        
        4D → 2D: (B, C, H, W) → (B*H*W, C)
        2D → 4D: (B*H*W, C) → (B, C, H, W)
        """
        if len(tensor.shape) == 4:
            # Image-like to vector-like: (B, C, H, W) → (B*H*W, C)
            B, C, H, W = tensor.shape
            # Reshape: first transpose to (B, H, W, C), then reshape
            tensor = tensor.transpose(0, 2, 3, 1)  # (B, H, W, C)
            tensor = tensor.reshape(B * H * W, C)   # (B*H*W, C)
        else:
            # Vector-like to image-like: (B*H*W, C) → (B, C, H, W)
            B, C, H, W = self.original_shape
            tensor = tensor.reshape(B, H, W, C)     # (B, H, W, C)
            tensor = tensor.transpose(0, 3, 1, 2)   # (B, C, H, W)
        
        return tensor
    
    def forward(self, input_tensor):
        """
        Forward pass for Batch Normalization.
        """
        # Check if convolutional (4D) or vector (2D)
        self.is_convolutional = len(input_tensor.shape) == 4
        
        if self.is_convolutional:
            # Store original shape for reformat back
            self.original_shape = input_tensor.shape
            # Reformat to 2D: (B, C, H, W) → (B*H*W, C)
            input_tensor = self.reformat(input_tensor)
        
        # Store input for backward pass
        self.input_tensor = input_tensor
        
        if self.testing_phase:
            # Testing: use running statistics
            self.input_normalized = (input_tensor - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            # Training: compute batch statistics
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            
            # Initialize running statistics with first batch
            if self.running_mean is None:
                self.running_mean = self.mean.copy()
                self.running_var = self.var.copy()
            else:
                # Update running statistics (moving average)
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            
            # Normalize using batch statistics
            self.input_normalized = (input_tensor - self.mean) / np.sqrt(self.var + self.eps)
        
        # Scale and shift: y = gamma * x_normalized + beta
        output = self.weights * self.input_normalized + self.bias
        
        if self.is_convolutional:
            # Reformat back to 4D: (B*H*W, C) → (B, C, H, W)
            output = self.reformat(output)
        
        return output
    
    def backward(self, error_tensor):
        """
        Backward pass for Batch Normalization.
        """
        if self.is_convolutional:
            # Reformat to 2D for computation
            error_tensor = self.reformat(error_tensor)
        
        # Gradient w.r.t. weights (gamma): sum over batch
        self._gradient_weights = np.sum(error_tensor * self.input_normalized, axis=0)
        
        # Gradient w.r.t. bias (beta): sum over batch
        self._gradient_bias = np.sum(error_tensor, axis=0)
        
        # Gradient w.r.t. input using helper function
        gradient_input = compute_bn_gradients(
            error_tensor, 
            self.input_tensor, 
            self.weights, 
            self.mean, 
            self.var,
            self.eps
        )
        
        # Update weights and bias if optimizers are set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        
        if self.is_convolutional:
            # Reformat back to 4D
            gradient_input = self.reformat(gradient_input)
        
        return gradient_input