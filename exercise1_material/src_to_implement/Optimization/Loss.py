import numpy as np

class Loss:
    def __init__(self, abc):
        pass

class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):

        self.prediction_tensor = prediction_tensor
        
        # Compute cross entropy loss
        # Add epsilon to avoid log(0)
        log_predictions = np.log(prediction_tensor + self.epsilon)
        
        # Element-wise multiply with labels and sum
        # Negative sign because cross entropy formula has negative
        loss = -np.sum(label_tensor * log_predictions)
        
        return loss
    
    def backward(self, label_tensor):
        # Gradient: -label / prediction
        error_tensor = -label_tensor / (self.prediction_tensor + self.epsilon)
        
        return error_tensor   