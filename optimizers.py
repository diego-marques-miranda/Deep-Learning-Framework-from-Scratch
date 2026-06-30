class SGD:
    """
    Stochastic Gradient Descent (Optimizer).
    The 'mechanic' responsible for updating the weights based on the gradient and learning rate.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        # We move the weight in the opposite direction of the gradient (descending the mountain)
        # The learning rate acts as a "brake" for taking short, safe steps
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases