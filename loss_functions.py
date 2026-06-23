import numpy as np

class mse:
    def __init__(self):
        self.output = None

    def forward(self, y_pred, y_real):
        self.output = np.mean((y_pred - y_real) ** 2)

        return self.output
    
    def backward(self, y_pred, y_real):
        samples = len(y_pred)
        self.dinputs = (2 / samples) * (y_pred - y_real)

        return self.dinputs