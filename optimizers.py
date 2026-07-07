from abc import ABC, abstractmethod    

class Base_Optimizer(ABC):
    """
    Abstract base class for optimizers.
    
    Defines the contract for all optimization algorithms in the framework,
    ensuring they implement the weight update logic.
    """
    @abstractmethod
    def update(self, layer):
        """
        Performs the weight and bias update for a given layer.
        
        Args:
            layer (object): The layer instance containing weights, 
                           biases, and their respective gradients.
        """
        pass

class SGD(Base_Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    The 'mechanic' responsible for updating the weights based on the 
    calculated gradients and the learning rate.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initializes the SGD optimizer.
        
        Args:
            learning_rate (float): The step size used for weight updates.
        """
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Updates the weights and biases of the provided layer using SGD.
        
        Args:
            layer (object): The layer to update. Must have 'weights', 
                           'biases', 'dweights', and 'dbiases' attributes.
        """
        # We move the weight in the opposite direction of the gradient 
        # (descending the mountain).
        # The learning rate acts as a "brake" for taking short, safe steps.
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases