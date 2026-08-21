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

class CrossEntropy(Base_Loss_Function):
    """
    Categorical Cross-Entropy loss function.
    
    Computes the negative log-likelihood of the true class labels given 
    the predicted probabilities. Used primarily in classification tasks.
    Assumes y_real contains sparse labels (class indices), not one-hot encoded vectors.
    """
    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, y_pred, y_real):
        """
        Performs the forward pass of the Cross-Entropy function.
        
        Args:
            y_pred (np.ndarray): Predicted probabilities from the model.
            y_real (np.ndarray): Ground truth target values (class indices).
            
        Returns:
            float: The calculated categorical cross-entropy loss.
        """
        # Extracts the predicted probability for the correct target class for each sample in the batch
        prob = y_pred[np.arange(len(y_pred)), y_real]

        self.output = (-np.log(prob + 1e-15)).mean()

        return self.output
    
    def backward(self, y_pred, y_real):
        """
        Performs the backward pass to calculate the gradient of the loss 
        with respect to the model outputs (predictions).
        
        Args:
            y_pred (np.ndarray): Predicted probabilities from the model.
            y_real (np.ndarray): Ground truth target values (class indices).
            
        Returns:
            np.ndarray: The gradient of the loss with respect to inputs.
        """
        # Initialize gradients array with the same shape as predictions
        self.dinputs = np.zeros_like(y_pred)

        # Extract probabilities of the correct classes
        prob = y_pred[np.arange(len(y_pred)), y_real]

        # The derivative of -log(x) with respect to x is -1/x.
        self.dinputs[np.arange(len(y_pred)), y_real] = -1 / (prob + 1e-15)

        self.dinputs /= len(y_pred)

        return self.dinputs