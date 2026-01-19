import numpy as np

class Constant:

    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        # All weights set to constant value
        return np.full(weights_shape, self.constant_value)

class UniformRandom:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # Random values from uniform distribution [0, 1)
        return np.random.uniform(0, 1, weights_shape)

class Xavier:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # Xavier/Glorot initialization
        # Good for tanh and sigmoid activations
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)

class He:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # He initialization
        # Good for ReLU activations
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)