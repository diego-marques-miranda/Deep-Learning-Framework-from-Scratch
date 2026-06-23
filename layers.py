import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_nodes):
        rng = np.random.default_rng()

        self.weights = rng.standard_normal((n_inputs, n_nodes)) * np.sqrt(2.0 / n_inputs)

        self.biases = np.zeros((1, n_nodes))

        self.inputs = None
        self.output = None

        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs