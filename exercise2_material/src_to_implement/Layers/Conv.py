import numpy as np
from scipy import signal
from Layers.Base import BaseLayer
from copy import deepcopy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        
        # Store parameters
        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape
        
        # Handle stride (can be single value or tuple)
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = stride_shape
        
        # Determine if 1D or 2D convolution
        # 1D: convolution_shape = [c, m]
        # 2D: convolution_shape = [c, m, n]
        self.is_1d = len(convolution_shape) == 2
        
        # Initialize weights: (num_kernels, channels, m) or (num_kernels, channels, m, n)
        weights_shape = (num_kernels,) + tuple(convolution_shape)
        self.weights = np.random.uniform(0, 1, weights_shape)
        
        # Initialize bias: one per kernel
        self.bias = np.random.uniform(0, 1, num_kernels)
        
        # Gradient storage
        self._gradient_weights = None
        self._gradient_bias = None
        
        # Optimizer (will be set externally)
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        # Calculate fan_in and fan_out
        if self.is_1d:
            # 1D: convolution_shape = [c, m]
            channels = self.convolution_shape[0]
            kernel_size = self.convolution_shape[1]
        else:
            # 2D: convolution_shape = [c, m, n]
            channels = self.convolution_shape[0]
            kernel_size = self.convolution_shape[1] * self.convolution_shape[2]
        
        fan_in = channels * kernel_size
        fan_out = self.num_kernels * kernel_size
        
        # Initialize weights
        weights_shape = (self.num_kernels,) + tuple(self.convolution_shape)
        self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        
        # Initialize bias
        bias_shape = (self.num_kernels,)
        self.bias = bias_initializer.initialize(bias_shape, 1, self.num_kernels)

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self._bias_optimizer = deepcopy(value)
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    def forward(self, input_tensor):
        # Store for backward pass
        self.input_tensor = input_tensor
        
        batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]
        
        if self.is_1d:
            # 1D convolution
            input_width = input_tensor.shape[2]
            
            # Calculate output size with padding (same convolution)
            # We need to pad to handle boundaries
            kernel_width = self.convolution_shape[1]
            
            # Pad input for 'same' convolution
            pad_width = kernel_width // 2
            pad_right = kernel_width - 1 - pad_width
            
            padded_input = np.pad(input_tensor, 
                                  ((0, 0), (0, 0), (pad_width, pad_right)), 
                                  mode='constant')
            
            # Calculate output width
            out_width = int(np.ceil(input_width / self.stride_shape[0]))
            
            # Initialize output
            output = np.zeros((batch_size, self.num_kernels, out_width))
            
            # Perform convolution for each batch and kernel
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    conv_result = np.zeros(input_width)
                    for c in range(num_channels):
                        # correlate is convolution without flipping the kernel
                        conv_result += signal.correlate(padded_input[b, c], 
                                                        self.weights[k, c], 
                                                        mode='valid')
                    # Apply stride
                    output[b, k] = conv_result[::self.stride_shape[0]] + self.bias[k]
        
        else:
            # 2D convolution
            input_height = input_tensor.shape[2]
            input_width = input_tensor.shape[3]
            
            kernel_height = self.convolution_shape[1]
            kernel_width = self.convolution_shape[2]
            
            # Padding for 'same' convolution
            pad_top = kernel_height // 2
            pad_bottom = kernel_height - 1 - pad_top
            pad_left = kernel_width // 2
            pad_right = kernel_width - 1 - pad_left
            
            padded_input = np.pad(input_tensor,
                                  ((0, 0), (0, 0), 
                                   (pad_top, pad_bottom), 
                                   (pad_left, pad_right)),
                                  mode='constant')
            
            # Calculate output size
            stride_y = self.stride_shape[0]
            stride_x = self.stride_shape[1] if len(self.stride_shape) > 1 else self.stride_shape[0]
            
            out_height = int(np.ceil(input_height / stride_y))
            out_width = int(np.ceil(input_width / stride_x))
            
            # Initialize output
            output = np.zeros((batch_size, self.num_kernels, out_height, out_width))
            
            # Perform convolution
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    conv_result = np.zeros((input_height, input_width))
                    for c in range(num_channels):
                        conv_result += signal.correlate2d(padded_input[b, c],
                                                          self.weights[k, c],
                                                          mode='valid')
                    # Apply stride
                    output[b, k] = conv_result[::stride_y, ::stride_x] + self.bias[k]
        
        return output
    
    def backward(self, error_tensor):
        # Initialize gradients
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros(self.num_kernels)
        
        batch_size = self.input_tensor.shape[0]
        num_channels = self.input_tensor.shape[1]
        
        # Gradient w.r.t bias: sum over batch and spatial dimensions
        if self.is_1d:
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))
        else:
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        
        # Initialize error for previous layer
        error_prev = np.zeros_like(self.input_tensor)
        
        if self.is_1d:
            input_width = self.input_tensor.shape[2]
            kernel_width = self.convolution_shape[1]
            
            # Upsample error tensor if stride > 1
            if self.stride_shape[0] > 1:
                upsampled_error = np.zeros((batch_size, self.num_kernels, input_width))
                upsampled_error[:, :, ::self.stride_shape[0]] = error_tensor
                error_tensor = upsampled_error
            
            # Padding
            pad_width = kernel_width // 2
            pad_right = kernel_width - 1 - pad_width
            
            padded_input = np.pad(self.input_tensor,
                                  ((0, 0), (0, 0), (pad_width, pad_right)),
                                  mode='constant')
            
            # Compute gradients
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        # Gradient w.r.t weights
                        self._gradient_weights[k, c] += signal.correlate(
                            padded_input[b, c], error_tensor[b, k], mode='valid')
                        
                        # Gradient w.r.t input (full convolution with flipped kernel)
                        error_prev[b, c] += signal.convolve(
                            error_tensor[b, k], self.weights[k, c], mode='same')
        
        else:
            input_height = self.input_tensor.shape[2]
            input_width = self.input_tensor.shape[3]
            kernel_height = self.convolution_shape[1]
            kernel_width = self.convolution_shape[2]
            
            stride_y = self.stride_shape[0]
            stride_x = self.stride_shape[1] if len(self.stride_shape) > 1 else self.stride_shape[0]
            
            # Upsample error tensor if stride > 1
            if stride_y > 1 or stride_x > 1:
                upsampled_error = np.zeros((batch_size, self.num_kernels, input_height, input_width))
                upsampled_error[:, :, ::stride_y, ::stride_x] = error_tensor
                error_tensor = upsampled_error
            
            # Padding
            pad_top = kernel_height // 2
            pad_bottom = kernel_height - 1 - pad_top
            pad_left = kernel_width // 2
            pad_right = kernel_width - 1 - pad_left
            
            padded_input = np.pad(self.input_tensor,
                                  ((0, 0), (0, 0),
                                   (pad_top, pad_bottom),
                                   (pad_left, pad_right)),
                                  mode='constant')
            
            # Compute gradients
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(num_channels):
                        # Gradient w.r.t weights
                        self._gradient_weights[k, c] += signal.correlate2d(
                            padded_input[b, c], error_tensor[b, k], mode='valid')
                        
                        # Gradient w.r.t input
                        error_prev[b, c] += signal.convolve2d(
                            error_tensor[b, k], self.weights[k, c], mode='same')
        
        # Update weights if optimizer is set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        
        return error_prev
