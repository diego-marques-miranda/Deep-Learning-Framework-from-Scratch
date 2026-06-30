import numpy as np

class mse:
    """
    Mean Squared Error.
    Quadratic penalty for the difference between the prediction and the actual target.
    """
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_real):
        # Difference squared to avoid negative values and heavily penalize large errors
        return np.mean((y_pred - y_real) ** 2)
    
    def backward(self, y_pred, y_real):
        samples = len(y_pred)
        # The derivative of x^2 is 2x. We normalize by the number of samples (samples) 
        # so that large batch gradients don't explode the network.
        self.dinputs = (2 / samples) * (y_pred - y_real)
        return self.dinputs