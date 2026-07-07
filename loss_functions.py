import numpy as np
from abc import ABC, abstractmethod    

class Base_Loss_Function(ABC):
    """
    Abstract base class for loss functions.
    
    Defines the contract for all loss calculations in the framework,
    ensuring they implement forward and backward passes.
    """
    @abstractmethod
    def forward(self, y_pred, y_real):
        """
        Performs the forward pass to calculate the loss.
        
        Args:
            y_pred (np.ndarray): Predicted values from the model.
            y_real (np.ndarray): Ground truth target values.
        """
        pass

    @abstractmethod
    def backward(self, y_pred, y_real):
        """
        Performs the backward pass to calculate gradients.
        
        Args:
            y_pred (np.ndarray): Predicted values from the model.
            y_real (np.ndarray): Ground truth target values.
        """
        pass

class Mse(Base_Loss_Function):
    """
    Mean Squared Error (MSE) loss function.
    
    Computes the quadratic penalty for the difference between the 
    prediction and the actual target. Used primarily in regression tasks.
    """
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_real):
        """
        Performs the forward pass of the MSE function.
        
        Args:
            y_pred (np.ndarray): Predicted values from the model.
            y_real (np.ndarray): Ground truth target values.
            
        Returns:
            float: The calculated mean squared error.
        """
        # Difference squared to avoid negative values and heavily penalize large errors
        return np.mean((y_pred - y_real) ** 2)
    
    def backward(self, y_pred, y_real):
        """
        Performs the backward pass to calculate the gradient of the loss 
        with respect to the model outputs.
        
        Args:
            y_pred (np.ndarray): Predicted values from the model.
            y_real (np.ndarray): Ground truth target values.
            
        Returns:
            np.ndarray: The gradient of the loss with respect to inputs.
        """
        samples = len(y_pred)
        # The derivative of x^2 is 2x. We normalize by the number of samples (samples) 
        # so that large batch gradients don't explode the network.
        self.dinputs = (2 / samples) * (y_pred - y_real)
        return self.dinputs