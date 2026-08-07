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
    def __init__(self, learning_rate=0.01, beta=0.9, debug=False, debug_interval=100):
        self.learning_rate = learning_rate
        self.beta = beta
        self.debug = debug
        self.debug_interval = debug_interval
        self.state = {}
        self.step_count = 0

    def update(self, layer):
        # Inicializa o estado se a camada ainda não existir no dict
        if layer not in self.state.keys():
            if getattr(layer, 'dweights', None) is not None:
                self.state[layer] = {
                    "weights": {
                        "cache": np.zeros_like(layer.dweights)
                    },
                    "biases": {
                        "cache": np.zeros_like(layer.dbiases)
                    }
                }

        # Garante que a camada possui estado inicializado
        if layer not in self.state:
            return

        # 1. Atualização do Cache (Média Móvel dos Quadrados dos Gradientes)
        self.state[layer]['weights']['cache'] = (
            self.beta * self.state[layer]['weights']['cache'] + (1 - self.beta) * (layer.dweights ** 2)
        )
        self.state[layer]['biases']['cache'] = (
            self.beta * self.state[layer]['biases']['cache'] + (1 - self.beta) * (layer.dbiases ** 2)
        )

        # 2. Cálculos intermediários para os Pesos
        w_cache = self.state[layer]['weights']['cache']
        normalized_gradient_w = layer.dweights / (np.sqrt(w_cache) + 1e-8)
        effective_step_w = self.learning_rate * normalized_gradient_w

        # 3. Atualização dos Parâmetros
        layer.weights -= effective_step_w
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(self.state[layer]['biases']['cache']) + 1e-8)

        # 4. Disparo de Debug (imprime informações do primeiro peso da camada)
        self.step_count += 1
        if self.debug and (self.step_count % self.debug_interval == 0):
            # Extrai o primeiro elemento para amostragem
            g = layer.dweights.flat[0]
            c = w_cache.flat[0]
            ng = normalized_gradient_w.flat[0]
            es = effective_step_w.flat[0]

            print(f"--- [RMSprop Debug - Step {self.step_count}] ---")
            print(f"Peso 1:")
            print(f"  gradient            = {g:.8f}")
            print(f"  cache               = {c:.8f}")
            print(f"  normalized_gradient = {ng:.8f}")
            print(f"  effective_step      = {es:.8f}\n")