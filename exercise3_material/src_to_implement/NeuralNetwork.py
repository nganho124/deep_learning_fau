import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        # Phase setting: False = training, True = testing
        self._phase = False  

    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, value):
        """Set the phase for all layers in the network."""
        self._phase = value
        for layer in self.layers:
            layer.testing_phase = value

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        # Calculate data loss
        data_loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        
        # Add regularization loss from all trainable layers
        regularization_loss = self._calculate_regularization_loss()
        
        return data_loss + regularization_loss
    
    def _calculate_regularization_loss(self):
        """Calculate the total regularization loss from all trainable layers."""
        reg_loss = 0
        
        for layer in self.layers:
            if layer.trainable:
                # Check if layer has optimizer with regularizer
                if hasattr(layer, 'optimizer') and layer.optimizer is not None:
                    if layer.optimizer.regularizer is not None:
                        reg_loss += layer.optimizer.regularizer.norm(layer.weights)
                
                # Check if layer has its own calculate_regularization_loss method (for RNN)
                if hasattr(layer, 'calculate_regularization_loss'):
                    reg_loss += layer.calculate_regularization_loss()
        
        return reg_loss
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if self.weights_initializer is not None and self.bias_initializer is not None:
                layer.initialize(self.weights_initializer, self.bias_initializer)
        
        # Set the layer's phase to match the network's current phase
        layer.testing_phase = self._phase
        
        self.layers.append(layer)

    def train(self, iterations):
        # Set to training phase
        self.phase = False  # <-- IMPORTANT: Set all layers to training mode
        
        for _ in range(iterations):
            loss_value = self.forward()
            self.loss.append(loss_value)
            self.backward()

    def test(self, input_tensor):
        # Set to testing phase
        self.phase = True  # <-- IMPORTANT: Set all layers to testing mode
        
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        return input_tensor