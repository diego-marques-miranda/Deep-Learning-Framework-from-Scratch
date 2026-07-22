import numpy as np
import pickle

class Model:
    """
    The Core Framework.
    Manages layers, compiles configurations, orchestrates training (fit), and makes predictions (predict).
    """
    def __init__(self, scaler):
        self.layers = []
        self.loss_func = None
        self.optimizer = None

        self.train_loss = None
        self.val_loss = None

        self.scaler = scaler

    def add(self, layer):
        """Appends a layer to the neural network architecture."""
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """Configures the loss function and optimizer for training."""
        self.loss_func = loss
        self.optimizer = optimizer

    def fit(self, train_dataloader, test_dataloader, epochs, train_metrics=[], test_metrics=[]):
        """Orchestrates the training and evaluation loop across epochs."""
        for epoch in range(epochs):
            # Reset metric states at the start of each epoch
            for m in train_metrics: m.reset()
            for m in test_metrics: m.reset()

            # Batch Training
            for X_batch, y_batch in train_dataloader:
                output = X_batch

                # Forward pass
                for layer in self.layers:
                    output = layer.forward(output)

                self.train_loss = self.loss_func.forward(output, y_batch)

                # Backward pass & parameter optimization
                gradient = self.loss_func.backward(output, y_batch)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)
                    if hasattr(layer, 'weights'):
                        self.optimizer.update(layer)

                # Update training metrics
                for m in train_metrics:
                    m.update(y_batch, output)

            # Test / Evaluation loop
            for X_test, y_test in test_dataloader:
                y_pred = self.predict(X_test)

                # Update test metrics
                for m in test_metrics:
                    m.update(y_test, y_pred)

            # Compile log outputs
            train_epoch_metrics = [f"{m.__class__.__name__}: {m.result():.2f}%" for m in train_metrics]
            test_epoch_metrics = [f"{m.__class__.__name__}: {m.result():.2f}%" for m in test_metrics]

            # Print progress selectively
            if epoch % (epochs // 20) == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} | Training: {' | '.join(train_epoch_metrics)} | Test: {' | '.join(test_epoch_metrics)}")

    def predict(self, X):
        """Performs a forward pass through all layers to generate predictions."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def save(self, path):
        """Serializes layer parameters and scaler state into a pickle file."""
        parameters = []

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                parameters.append([layer.weights, layer.biases])

        data_to_save = {
            'parameters': parameters,
            'scaler': self.scaler
        }

        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)
            
        print(f"Model saved to {path}")

    def load(self, path):
        """Restores layer weights/biases and scaler state from a pickle file."""
        with open(path, 'rb') as file:
            data = pickle.load(file)

        self.scaler = data['scaler']
        i = 0

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = data['parameters'][i][0]
                layer.biases = data['parameters'][i][1]
                i += 1
                
        print(f"Model loaded from {path}")  