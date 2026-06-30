import numpy as np

class Activation_ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    Introduces non-linearity by outputting the input directly if positive, otherwise zero.
    """
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        # Cache inputs for derivative computation during backpropagation
        self.inputs = inputs
        
        # Forward pass computation: f(x) = max(0, x)
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, dvalues):
        # Create a copy to prevent modifying the original gradients by reference
        dinputs = dvalues.copy()
        
        # Zero out gradients where the original input was negative or zero (ReLU derivative)
        dinputs[self.inputs <= 0] = 0

        return dinputs

class Activation_Linear:
    """
    Linear (Identity) activation function.
    Returns the input unchanged. Standard activation for regression output layers.
    """
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        # Cache inputs to maintain API consistency across activation layers
        self.inputs = inputs
        
        # Identity transformation: f(x) = x
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        # The derivative of f(x) = x is 1. Gradients are passed through unmodified.
        return dvalues