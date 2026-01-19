import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        # Pooling has no trainable parameters
        
        # stride_shape = (stride_y, stride_x)
        self.stride_shape = stride_shape
        
        # pooling_shape = (pool_height, pool_width)
        self.pooling_shape = pooling_shape
    
    def forward(self, input_tensor):
        # Store input shape for backward
        self.input_tensor = input_tensor
        
        batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]
        
        pool_height = self.pooling_shape[0]
        pool_width = self.pooling_shape[1]
        stride_y = self.stride_shape[0]
        stride_x = self.stride_shape[1]
        
        # Calculate output size (valid padding)
        out_height = (input_height - pool_height) // stride_y + 1
        out_width = (input_width - pool_width) // stride_x + 1
        
        # Initialize output
        output = np.zeros((batch_size, num_channels, out_height, out_width))
        
        # Store max indices for backward pass
        self.max_indices = np.zeros((batch_size, num_channels, out_height, out_width, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(num_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # Calculate window position
                        h_start = h * stride_y
                        h_end = h_start + pool_height
                        w_start = w * stride_x
                        w_end = w_start + pool_width
                        
                        # Extract window
                        window = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        
                        # Find max value
                        output[b, c, h, w] = np.max(window)
                        
                        # Store position of max value (relative to window)
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        # Convert to absolute position in input
                        self.max_indices[b, c, h, w, 0] = h_start + max_idx[0]
                        self.max_indices[b, c, h, w, 1] = w_start + max_idx[1]
        
        return output
    
    def backward(self, error_tensor):
        batch_size = self.input_tensor.shape[0]
        num_channels = self.input_tensor.shape[1]
        
        out_height = error_tensor.shape[2]
        out_width = error_tensor.shape[3]
        
        # Initialize error for previous layer (same shape as input)
        error_prev = np.zeros_like(self.input_tensor)
        
        # Distribute gradients to max positions
        for b in range(batch_size):
            for c in range(num_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # Get the position that had the max value
                        max_h = self.max_indices[b, c, h, w, 0]
                        max_w = self.max_indices[b, c, h, w, 1]
                        
                        # Add gradient to that position
                        # (use += because multiple outputs might map to same input with overlapping windows)
                        error_prev[b, c, max_h, max_w] += error_tensor[b, c, h, w]
        
        return error_prev