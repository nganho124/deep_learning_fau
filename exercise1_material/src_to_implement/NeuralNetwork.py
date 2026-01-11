import copy

class NeuralNetwork:

    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):

        input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        output = self.loss_layer.forward(input_tensor, self.label_tensor)
        
        return output
    
    def backward(self):

        # 1. Start backpropagation from loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        # 2. Propagate backward through all layers (reversed order!)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        # If layer is trainable, give it a deep copy of optimizer
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        
        # Add layer to network
        self.layers.append(layer)


    def train(self, iterations):
        for _ in range(iterations):
            # Forward pass - get loss
            loss_value = self.forward()
            
            # Store loss
            self.loss.append(loss_value)
            
            # Backward pass - update weights
            self.backward()


    def test(self, input_tensor):
        # Pass through all layers (no loss computation)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        # Return predictions (e.g., SoftMax probabilities)
        return input_tensor
