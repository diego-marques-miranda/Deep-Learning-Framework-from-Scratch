from abc import ABC, abstractmethod
import numpy as np

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

class SGD_Momentum(Base_Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with Momentum.
    
    Accelerates SGD in the relevant direction and dampens oscillations by 
    accumulating a moving average of past gradients (velocity), behaving 
    like a heavy ball rolling down a hill.
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Initializes the SGD with Momentum optimizer.
        
        Args:
            learning_rate (float): The step size used for weight updates.
            beta (float): The momentum decay hyperparameter (typically around 0.9). 
                          Determines how much weight is given to previous velocity.
        """
        self.learning_rate = learning_rate
        self.beta = beta
        # State tracking dictionary to maintain historical velocity per layer
        self.state = {}

    def update(self, layer):
        """
        Updates the weights and biases of the provided layer using SGD with Momentum.
        
        Args:
            layer (object): The layer to update. Must have 'weights', 
                            'biases', 'dweights', and 'dbiases' attributes.
        """
        # Initialize velocity buffers for this layer if encountered for the first time
        if layer not in self.state.keys():
            if layer.dweights is not None:
                self.state[layer] = {
                    "weights": {
                        "velocity": np.zeros_like(layer.dweights)
                    },
                    "biases": {
                        "velocity": np.zeros_like(layer.dbiases)
                    }
                }

        # Update velocity (exponential moving average of gradients)
        # Beta dampens past velocity, while (1 - beta) incorporates the current gradient force
        self.state[layer]['weights']['velocity'] = (
            self.beta * self.state[layer]['weights']['velocity'] + (1 - self.beta) * layer.dweights
        )
        self.state[layer]['biases']['velocity'] = (
            self.beta * self.state[layer]['biases']['velocity'] + (1 - self.beta) * layer.dbiases
        )

        # Update weights and biases using the accumulated momentum/velocity vector
        layer.weights -= self.learning_rate * self.state[layer]['weights']['velocity']
        layer.biases -= self.learning_rate * self.state[layer]['biases']['velocity']

class RMSprop(Base_Optimizer):
    """
    Root Mean Square Propagation (RMSprop) optimizer.
    
    Adapts the learning rate for each parameter by dividing by an exponentially 
    decaying average of squared gradients, helping to equalize step sizes across dimensions.
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Initializes the RMSprop optimizer.
        
        Args:
            learning_rate (float): The step size used for weight updates.
            beta (float): The decay rate for the moving average of squared gradients (typically 0.9).
        """
        self.learning_rate = learning_rate
        self.beta = beta
        # State tracking dictionary to maintain squared gradient cache per layer
        self.state = {}

    def update(self, layer):
        """
        Updates the weights and biases of the provided layer using RMSprop.
        
        Args:
            layer (object): The layer to update. Must have 'weights', 
                            'biases', 'dweights', and 'dbiases' attributes.
        """
        # Initialize cache buffers for this layer if encountered for the first time
        if layer not in self.state.keys():
            if layer.dweights is not None:
                self.state[layer] = {
                    "weights": {
                        "cache": np.zeros_like(layer.dweights)
                    },
                    "biases": {
                        "cache": np.zeros_like(layer.dbiases)
                    }
                }

        # Update cache (exponential moving average of squared gradients)
        self.state[layer]['weights']['cache'] = (
            self.beta * self.state[layer]['weights']['cache'] + (1 - self.beta) * (layer.dweights ** 2)
        )
        self.state[layer]['biases']['cache'] = (
            self.beta * self.state[layer]['biases']['cache'] + (1 - self.beta) * (layer.dbiases ** 2)
        )

        # Update weights and biases by scaling learning rate inversely to gradient magnitudes
        layer.weights -= self.learning_rate * layer.dweights / (self.state[layer]['weights']['cache'] ** (1/2) + 1e-8)
        layer.biases -= self.learning_rate * layer.dbiases / (self.state[layer]['biases']['cache'] ** (1/2) + 1e-8)

class Adam(Base_Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    
    Combines the ideas of Momentum (first moment) and RMSprop (second moment) 
    with bias correction to adjust individual learning rates per parameter.
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        """
        Initializes the Adam optimizer.
        
        Args:
            learning_rate (float): The step size used for weight updates.
            beta1 (float): Exponential decay rate for the first moment estimates (typically 0.9).
            beta2 (float): Exponential decay rate for the second moment estimates (typically 0.999).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        # State tracking dictionary to maintain time step and moment estimates per layer
        self.state = {}

    def update(self, layer):
        """
        Updates the weights and biases of the provided layer using Adam.
        
        Args:
            layer (object): The layer to update. Must have 'weights', 
                            'biases', 'dweights', and 'dbiases' attributes.
        """
        # Initialize state buffers and step counter for this layer if encountered for the first time
        if layer not in self.state.keys():
            if layer.dweights is not None:
                self.state[layer] = {
                    "t": 0,
                    "weights": {
                        "first_moment": np.zeros_like(layer.dweights),
                        "second_moment": np.zeros_like(layer.dweights)
                    },
                    "biases": {
                        "first_moment": np.zeros_like(layer.dbiases),
                        "second_moment": np.zeros_like(layer.dbiases)
                    }
                }

        # Increment time step for bias correction
        self.state[layer]["t"] += 1
        t = self.state[layer]["t"]

        # Update first moment estimates (momentum/velocity)
        self.state[layer]['weights']['first_moment'] = (
            self.beta1 * self.state[layer]['weights']['first_moment'] + (1 - self.beta1) * layer.dweights
        )
        self.state[layer]['biases']['first_moment'] = (
            self.beta1 * self.state[layer]['biases']['first_moment'] + (1 - self.beta1) * layer.dbiases
        )

        # Update second moment estimates (uncentered variance/RMSprop cache)
        self.state[layer]['weights']['second_moment'] = (
            self.beta2 * self.state[layer]['weights']['second_moment'] + (1 - self.beta2) * (layer.dweights ** 2)
        )
        self.state[layer]['biases']['second_moment'] = (
            self.beta2 * self.state[layer]['biases']['second_moment'] + (1 - self.beta2) * (layer.dbiases ** 2)
        )

        # Compute bias-corrected first moment estimates to counter initial zero-bias
        weight_first_corrected = self.state[layer]['weights']['first_moment'] / (1 - self.beta1 ** t)
        bias_first_corrected = self.state[layer]['biases']['first_moment'] / (1 - self.beta1 ** t)

        # Compute bias-corrected second moment estimates
        weight_second_corrected = self.state[layer]['weights']['second_moment'] / (1 - self.beta2 ** t)
        bias_second_corrected = self.state[layer]['biases']['second_moment'] / (1 - self.beta2 ** t)

        # Update weights and biases using corrected moment estimates
        layer.weights -= self.learning_rate * weight_first_corrected / (np.sqrt(weight_second_corrected) + 1e-8)
        layer.biases -= self.learning_rate * bias_first_corrected / (np.sqrt(bias_second_corrected) + 1e-8)