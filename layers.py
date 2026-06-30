import numpy as np

class Layer_Dense:
    """
    Dense Layer (Fully Connected).
    Each neuron in this layer connects to all received inputs.
    """
    def __init__(self, n_inputs, n_nodes, activation_func=None):
        rng = np.random.default_rng()

        # He Initialization
        # Multiplying by the square root of (2 / n_inputs) prevents exploding/vanishing gradients
        self.weights = rng.standard_normal((n_inputs, n_nodes)) * np.sqrt(2.0 / n_inputs)
        
        # Biases are initialized with zeros to not influence the output right away
        self.biases = np.zeros((1, n_nodes))

        self.inputs = None
        self.output = None

        self.dweights = None
        self.dbiases = None
        self.dinputs = None

        # Activation function injected directly into the layer
        self.activation_func = activation_func

    def forward(self, inputs):
        # Save inputs to use during the derivative calculation (backward pass)
        self.inputs = inputs

        # Matrix multiplication (Dot Product) added to the biases
        self.output = np.dot(inputs, self.weights) + self.biases

        # If there is an activation function, pass the result through it
        if self.activation_func:
            self.output = self.activation_func.forward(self.output)

        return self.output
    
    def backward(self, dvalues):
        # If there's an activation, the gradient (dvalues) passes through the reverse filter first
        if self.activation_func:
            dvalues = self.activation_func.backward(dvalues)

        # Gradient of weights: proportion of the error based on the input
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # Gradient of biases: sum of the error from each neuron
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient to be passed to the previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs