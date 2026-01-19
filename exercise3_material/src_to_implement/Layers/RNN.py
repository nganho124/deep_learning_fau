import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
import copy


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize Elman RNN layer.
        """
        # Create internal layers BEFORE calling super().__init__()
        # because super().__init__() sets self.weights = None
        # which triggers the setter that needs fc_hidden
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create internal layers FIRST
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh = TanH()
        self.fc_output = FullyConnected(hidden_size, output_size)
        
        # NOW call super().__init__() - it's safe because fc_hidden exists
        super().__init__()
        self.trainable = True
        
        # Initialize hidden state with zeros
        self.hidden_state = np.zeros(hidden_size)
        
        # Memorize flag
        self._memorize = False
        
        # Optimizer storage
        self._optimizer = None
        
        # Gradient storage
        self._gradient_weights = None
        
        # Storage for backward pass
        self.input_tensor = None
        self.hidden_states = []
        self.concat_inputs = []
    
    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, value):
        self._memorize = value
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self.fc_hidden.optimizer = copy.deepcopy(value)
        self.fc_output.optimizer = copy.deepcopy(value)
    
    @property
    def weights(self):
        """Return weights of fc_hidden."""
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        """Set weights of fc_hidden. Ignore None values."""
        if value is not None:
            self.fc_hidden.weights = value
    
    @property
    def gradient_weights(self):
        """Return gradient w.r.t. weights of fc_hidden."""
        return self._gradient_weights
    
    def initialize(self, weights_initializer, bias_initializer):
        """Initialize weights of internal fully connected layers."""
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
    
    def calculate_regularization_loss(self):
        """Calculate regularization loss from internal layers."""
        reg_loss = 0
        
        if self.fc_hidden.optimizer is not None:
            if self.fc_hidden.optimizer.regularizer is not None:
                reg_loss += self.fc_hidden.optimizer.regularizer.norm(self.fc_hidden.weights)
        
        if self.fc_output.optimizer is not None:
            if self.fc_output.optimizer.regularizer is not None:
                reg_loss += self.fc_output.optimizer.regularizer.norm(self.fc_output.weights)
        
        return reg_loss
    
    def forward(self, input_tensor):
        """
        Forward pass through RNN.
        """
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        output_tensor = np.zeros((batch_size, self.output_size))
        
        self.hidden_states = []
        self.concat_inputs = []
        
        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        
        h_prev = self.hidden_state.copy()
        
        for t in range(batch_size):
            x_t = input_tensor[t]
            
            concat_input = np.concatenate([x_t, h_prev])
            self.concat_inputs.append(concat_input)
            
            concat_input = concat_input.reshape(1, -1)
            
            fc_hidden_out = self.fc_hidden.forward(concat_input)
            h_t = self.tanh.forward(fc_hidden_out)
            h_t = h_t.flatten()
            
            self.hidden_states.append(h_t.copy())
            
            h_t_reshaped = h_t.reshape(1, -1)
            y_t = self.fc_output.forward(h_t_reshaped)
            output_tensor[t] = y_t.flatten()
            
            h_prev = h_t.copy()
        
        self.hidden_state = h_prev.copy()
        
        return output_tensor
    
    def backward(self, error_tensor):
        """
        Backward pass through RNN (BPTT).
        """
        batch_size = error_tensor.shape[0]
        
        error_prev = np.zeros((batch_size, self.input_size))
        error_h_next = np.zeros((1, self.hidden_size))
        
        self._gradient_weights = np.zeros_like(self.fc_hidden.weights)
        
        for t in reversed(range(batch_size)):
            error_y_t = error_tensor[t].reshape(1, -1)
            
            h_t = self.hidden_states[t].reshape(1, -1)
            self.fc_output.input_tensor = np.hstack([h_t, np.ones((1, 1))])
            
            error_h_from_output = self.fc_output.backward(error_y_t)
            
            error_h_t = error_h_from_output + error_h_next
            
            self.tanh.activation = h_t
            error_tanh = self.tanh.backward(error_h_t)
            
            concat_input = self.concat_inputs[t].reshape(1, -1)
            self.fc_hidden.input_tensor = np.hstack([concat_input, np.ones((1, 1))])
            
            error_concat = self.fc_hidden.backward(error_tanh)
            
            self._gradient_weights += self.fc_hidden.gradient_weights
            
            error_x_t = error_concat[0, :self.input_size]
            error_h_next = error_concat[0, self.input_size:].reshape(1, -1)
            
            error_prev[t] = error_x_t
        
        if self._optimizer is not None:
            self.fc_hidden.weights = self._optimizer.calculate_update(
                self.fc_hidden.weights, self._gradient_weights
            )
        
        return error_prev