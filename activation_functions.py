import numpy as np
from abc import ABC, abstractmethod    

class Base_Activation_Function(ABC):
    """
    Abstract base class for activation functions.
    
    Defines the contract for all activation components in the framework,
    ensuring they implement forward and backward passes.
    """
    @abstractmethod
    def forward(self, inputs):
        """
        Performs the forward pass.
        
        Args:
            inputs (np.ndarray): Input data from the previous layer.
        """
        pass

    @abstractmethod
    def backward(self, dvalues):
        """
        Performs the backward pass.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.
        """
        pass

class ReLU(Base_Activation_Function):
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
            
        Returns:
            np.ndarray: The activated input.
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
            
        Returns:
            np.ndarray: The gradient passed back to the previous layer.
        """
        # Create a copy to prevent in-place modification of incoming gradients
        self.dinputs = dvalues.copy()
        
        # Zero out gradients where input was non-positive (f'(x) = 0 for x <= 0)
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Leaky_ReLU(Base_Activation_Function):
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
        
        Returns:
            np.ndarray: The activated input.
        """
        self.inputs = inputs
        self.output = np.maximum(self.neg_slope * inputs, inputs)
        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate gradients.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.
            
        Returns:
            np.ndarray: The gradient passed back to the previous layer.
        """
        self.dinputs = dvalues.copy()
        
        # Apply the slope multiplier where inputs were non-positive (f'(x) = neg_slope)
        self.dinputs[self.inputs <= 0] *= self.neg_slope
        return self.dinputs

class Linear(Base_Activation_Function):
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
        
        Returns:
            np.ndarray: The input data unchanged.
        """
        self.inputs = inputs
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        """
        Performs the backward pass. Since the derivative of f(x) = x is 1,
        the gradient is passed through unchanged.
        
        Returns:
            np.ndarray: The unchanged gradient.
        """
        return dvalues

class Sigmoid(Base_Activation_Function):
    """
    Sigmoid activation function.

    Computes f(x) = 1 / (1 + e^(-x)). Maps input values to a range
    between 0 and 1, making it suitable for binary classification tasks.
    """
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass of the Sigmoid function.

        Args:
            inputs (np.ndarray): Input data from the previous layer.

        Returns:
            np.ndarray: The activated input, with values between 0 and 1.
        """
        self.inputs = inputs
        self.output = 1 / (1 + (np.e ** (-1 * inputs)))

        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate the gradient of the loss
        with respect to the inputs.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient passed back to the previous layer.
        """
        # Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
        self.dinputs = dvalues * (self.output * (1 - self.output))

        return self.dinputs
    
class Tanh(Base_Activation_Function):
    """
    Hyperbolic Tangent (Tanh) activation function.

    Computes f(x) = tanh(x). Maps input values to a range between -1 and 1,
    providing a zero-centered non-linear activation function.
    """
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass of the Tanh function.

        Args:
            inputs (np.ndarray): Input data from the previous layer.

        Returns:
            np.ndarray: The activated input, with values between -1 and 1.
        """
        self.inputs = inputs
        self.output = (
            np.e ** inputs - np.e ** (-1 * inputs)
        ) / (
            np.e ** inputs + np.e ** (-1 * inputs) + 1e-10
        )

        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate the gradient of the loss
        with respect to the inputs.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient passed back to the previous layer.
        """
        # Tanh derivative: f'(x) = 1 - f(x)^2
        self.dinputs = dvalues * (1 - self.output ** 2)

        return self.dinputs
    
class GELU(Base_Activation_Function):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Computes f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
    """
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass of the GELU function.

        Args:
            inputs (np.ndarray): Input data from the previous layer.

        Returns:
            np.ndarray: The activated input.
        """
        self.inputs = inputs

        cubic = inputs ** 3
        inner = inputs + 0.044715 * cubic
        tanh_input = np.sqrt(2 / np.pi) * inner
        
        self.tanh_val = np.tanh(tanh_input)
        
        self.output = 0.5 * inputs * (1 + self.tanh_val)

        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate the gradient of the loss
        with respect to the inputs for GELU.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient passed back to the previous layer.
        """
        x = self.inputs
        
        # Derivative of the inner function g(x) = sqrt(2/pi) * (x + 0.044715 * x^3)
        # g'(x) = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2) = sqrt(2/pi) * (1 + 0.134145 * x^2)
        g_prime = np.sqrt(2 / np.pi) * (1 + 0.134145 * (x ** 2))
        
        # Hyperbolic secant squared: sech^2(x) = 1 - tanh^2(x)
        sech_sq = 1 - (self.tanh_val ** 2)
        
        # Full GELU derivative:
        # f'(x) = 0.5 * (1 + tanh(g(x))) + 0.5 * x * sech^2(g(x)) * g'(x)
        derivative = 0.5 * (1 + self.tanh_val) + 0.5 * x * sech_sq * g_prime
        
        # Final gradient
        self.dinputs = dvalues * derivative

        return self.dinputs