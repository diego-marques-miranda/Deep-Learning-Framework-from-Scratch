import numpy as np

class Activation_ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Computes f(x) = max(0, x). Introduces non-linearity to the network 
    while maintaining computational efficiency.
    """
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass of the ReLU function.
        
        Args:
            inputs (np.ndarray): Input data from the previous layer.
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate the gradient of the loss 
        with respect to the inputs.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.
        """
        # Create a copy to prevent in-place modification of incoming gradients
        self.dinputs = dvalues.copy()
        
        # Zero out gradients where input was non-positive (f'(x) = 0 for x <= 0)
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Activation_Leaky_ReLU:
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    
    Computes f(x) = x if x > 0 else neg_slope * x. Prevents the "dying ReLU" 
    problem by allowing a small gradient flow for negative inputs.
    """
    def __init__(self, neg_slope=0.01):
        """
        Args:
            neg_slope (float): The multiplier for negative input values.
        """
        self.inputs = None
        self.output = None
        self.neg_slope = neg_slope

    def forward(self, inputs):
        """
        Performs the forward pass of the Leaky ReLU function.
        """
        self.inputs = inputs
        self.output = np.maximum(self.neg_slope * inputs, inputs)
        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate gradients.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.
        """
        self.dinputs = dvalues.copy()
        
        # Apply the slope multiplier where inputs were non-positive (f'(x) = neg_slope)
        self.dinputs[self.inputs <= 0] *= self.neg_slope
        return self.dinputs

class Activation_Linear:
    """
    Linear (Identity) activation function.
    
    Computes f(x) = x. Typically used in the output layer for regression 
    tasks where the network must predict unbounded real values.
    """
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass (identity mapping).
        """
        self.inputs = inputs
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        """
        Performs the backward pass. Since the derivative of f(x) = x is 1,
        the gradient is passed through unchanged.
        """
        return dvalues