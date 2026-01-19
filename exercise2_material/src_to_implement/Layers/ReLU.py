from Layers.Base import *
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        output = np.maximum(0, input_tensor)
        return output
    
    def backward(self, error_tensor):

        error_prev = error_tensor * (self.input_tensor > 0)

        return error_prev