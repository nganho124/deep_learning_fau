from Layers.Base import *

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):

        self.input_shape = input_tensor.shape

        batch_size = input_tensor.shape[0]

        output = input_tensor.reshape(batch_size, -1)

        return output
    
    def backward(self, error_tensor):

        error_prev = error_tensor.reshape(self.input_shape)

        return error_prev
