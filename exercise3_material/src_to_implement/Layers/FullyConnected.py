from Layers.Base import *
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.bias = np.random.uniform(0, 1, (1, output_size))
        # proteced member for optimizer
        self._optimizer = None
    
    def forward(self, input_tensor):
        # Add column of ones for bias
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        self.input_tensor = np.hstack((input_tensor, ones))  # (batch, input_size + 1)
        
        # Now weights include bias, so no separate + bias needed
        output = np.dot(self.input_tensor, self.weights)
        return output   
    
    # Getter property
    @property
    def optimizer(self):
        return self._optimizer
    
    # Setter property
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def backward(self, error_tensor):
        # 1. Compute gradient w.r.t weights
        gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self._gradient_weights = gradient_weights
        
        # 2. Compute gradient w.r.t bias
        gradient_bias = np.sum(error_tensor, axis=0, keepdims=True)
        
        # 3. Compute error tensor for previous layer
        error_prev = np.dot(error_tensor, self.weights[:-1, :].T)
        
        # 4. Update weights and bias if optimizer is set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, gradient_weights)
        
        return error_prev
    
    @property
    def gradient_weights(self):
        return self._gradient_weights

    
    def initialize(self, weights_initializer, bias_initializer):
        # Initialize weights (excluding bias row)
        weights = weights_initializer.initialize(
            (self.input_size, self.output_size),
            self.input_size,
            self.output_size
        )
        
        # Initialize bias
        bias = bias_initializer.initialize(
            (1, self.output_size),
            1,  # fan_in for bias
            self.output_size
        )
        
        # Combine weights and bias into single matrix
        self.weights = np.vstack((weights, bias))