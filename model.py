import numpy as np
import pickle

class Model:
    """
    The Core Framework.
    Manages layers, compiles configurations, orchestrates training (fit), and makes predictions (predict).
    """
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.optimizer = None

        self.train_loss = None
        self.val_loss = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_func = loss
        self.optimizer = optimizer

    def fit(self, X, y, epochs, batch_size, validation_split=0.0):
        validation_size = int(len(X) * validation_split)

        if validation_size > 0:
            X_train = X[validation_size : ]
            X_validation = X[ : validation_size]
            y_train = y[validation_size : ]
            y_validation = y[ : validation_size]
        else:
            X_train, X_validation = X, None
            y_train, y_validation = y, None

        for epoch in range(epochs):
            # Batch Training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                output = X_batch

                for layer in self.layers:
                    output = layer.forward(output)

                self.train_loss = self.loss_func.forward(output, y_batch)

                gradient = self.loss_func.backward(output, y_batch)

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)

                    if hasattr(layer, 'weights'):   # Updates the layer only if it has weights
                        self.optimizer.update(layer)

            # Full pass on training data to get precise MAPE
            train_predictions = self.predict(X_train)
            # We add 1e-8 to avoid ZeroDivisionError in case y_train has zeros
            train_mape = np.mean(np.abs((y_train - train_predictions) / (y_train + 1e-8))) * 100

            if validation_size > 0:
                val_predictions = self.predict(X_validation)
                self.val_loss = self.loss_func.forward(val_predictions, y_validation)
                val_mape = np.mean(np.abs((y_validation - val_predictions) / (y_validation + 1e-8))) * 100

                if epoch % 50 == 0 or epoch == epochs - 1:
                    print(f'Epoch: {epoch} | Train loss: {self.train_loss:.4f} - MAPE: {train_mape:.2f}% | Val loss: {self.val_loss:.4f} - Val MAPE: {val_mape:.2f}%')
            else:
                if epoch % 50 == 0 or epoch == epochs - 1:
                    print(f'Epoch: {epoch} | Train loss: {self.train_loss:.4f} - MAPE: {train_mape:.2f}%')

    def predict(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output
    
    def save(self, path):
        parameters = []

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                parameters.append([layer.weights, layer.biases])

        with open(path, 'wb') as file:
            pickle.dump(parameters, file)
            
        print(f"Model saved to {path}")

    def load(self, path):
        with open(path, 'rb') as file:
            parameters = pickle.load(file)

        i = 0

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = parameters[i][0]
                layer.biases = parameters[i][1]
                i += 1
                
        print(f"Model loaded from {path}")