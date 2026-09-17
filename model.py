import pickle
import copy


class Model:
    """
    The Core Framework.
    Manages layers, compiles configurations, orchestrates training (fit),
    and makes predictions (predict).
    """

    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.optimizer = None

        self.train_loss = None
        self.val_loss = None

        self.history = {}

        # Training state
        self.best_epoch = None
        self.best_value = None
        self.epochs_trained = 0
        self.stopped_early = False

    def add(self, layer):
        """Appends a layer to the neural network architecture."""
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """Configures the loss function and optimizer for training."""
        self.loss_func = loss
        self.optimizer = optimizer

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        epochs,
        train_metrics=None,
        val_metrics=None,
        early_stopping=False,
        monitor="val_loss",
        mode="min",
        patience=30,
        min_delta=0.0,
        restore_best_weights=True
    ):
        """Orchestrates the training and validation loop across epochs."""

        if train_metrics is None:
            train_metrics = []

        if val_metrics is None:
            val_metrics = []

        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")

        if patience < 0:
            raise ValueError("patience must be greater than or equal to 0")

        if min_delta < 0:
            raise ValueError("min_delta must be greater than or equal to 0")

        # History
        self.history = {
            "loss": [],
            "val_loss": []
        }

        for metric in train_metrics:
            self.history[metric.__class__.__name__] = []

        for metric in val_metrics:
            self.history[f"val_{metric.__class__.__name__}"] = []

        if monitor not in self.history:
            raise ValueError(
                f"Monitor '{monitor}' is not available in the training history."
            )

        # Reset training state
        self.best_epoch = None
        self.best_value = None
        self.epochs_trained = 0
        self.stopped_early = False

        best_weights = None
        patience_counter = 0

        print_interval = max(1, epochs // 20)

        for epoch in range(epochs):

            # Reset metric states at the start of each epoch
            for metric in train_metrics:
                metric.reset()

            for metric in val_metrics:
                metric.reset()

            # -------------------------
            # Training
            # -------------------------

            train_loss_sum = 0.0
            train_samples = 0

            for X_batch, y_batch in train_dataloader:

                output = X_batch

                # Forward pass
                for layer in self.layers:
                    output = layer.forward(output)

                batch_loss = self.loss_func.forward(output, y_batch)

                train_loss_sum += batch_loss * len(y_batch)
                train_samples += len(y_batch)

                # Backward pass
                gradient = self.loss_func.backward(output, y_batch)

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)

                    if hasattr(layer, "weights"):
                        self.optimizer.update(layer)

                # Update training metrics
                for metric in train_metrics:
                    metric.update(y_batch, output)

            self.train_loss = train_loss_sum / train_samples

            # -------------------------
            # Validation
            # -------------------------

            val_loss_sum = 0.0
            val_samples = 0

            for X_batch, y_batch in val_dataloader:

                y_pred = self.predict(X_batch)

                batch_loss = self.loss_func.forward(y_pred, y_batch)

                val_loss_sum += batch_loss * len(y_batch)
                val_samples += len(y_batch)

                for metric in val_metrics:
                    metric.update(y_batch, y_pred)

            self.val_loss = val_loss_sum / val_samples

            # -------------------------
            # Save history
            # -------------------------

            self.history["loss"].append(self.train_loss)
            self.history["val_loss"].append(self.val_loss)

            for metric in train_metrics:
                self.history[metric.__class__.__name__].append(
                    metric.result()
                )

            for metric in val_metrics:
                self.history[f"val_{metric.__class__.__name__}"].append(
                    metric.result()
                )

            # -------------------------
            # Early stopping / checkpoint
            # -------------------------

            if early_stopping:

                current_value = self.history[monitor][-1]

                if self.best_value is None:
                    improved = True
                elif mode == "min":
                    improved = current_value < (
                        self.best_value - min_delta
                    )
                else:
                    improved = current_value > (
                        self.best_value + min_delta
                    )

                if improved:
                    self.best_value = current_value
                    self.best_epoch = epoch + 1
                    patience_counter = 0

                    # Save a snapshot of trainable parameters
                    best_weights = []

                    for layer in self.layers:
                        if hasattr(layer, "weights"):
                            best_weights.append({
                                "weights": copy.deepcopy(layer.weights),
                                "biases": copy.deepcopy(layer.biases)
                            })

                else:
                    patience_counter += 1

            self.epochs_trained = epoch + 1

            # -------------------------
            # Print progress
            # -------------------------

            if epoch % print_interval == 0 or epoch == epochs - 1:

                train_epoch_metrics = [
                    f"{metric.__class__.__name__}: {metric.result():.2f}"
                    for metric in train_metrics
                ]

                val_epoch_metrics = [
                    f"{metric.__class__.__name__}: {metric.result():.2f}"
                    for metric in val_metrics
                ]

                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {self.train_loss:.4f} | "
                    f"{' | '.join(train_epoch_metrics)} | "
                    f"Val Loss: {self.val_loss:.4f} | "
                    f"{' | '.join(val_epoch_metrics)}"
                )

            # -------------------------
            # Early stopping condition
            # -------------------------

            if early_stopping and patience_counter >= patience:

                self.stopped_early = True

                print(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best {monitor}: {self.best_value:.4f} "
                    f"at epoch {self.best_epoch}."
                )

                break

        # -------------------------
        # Restore best weights
        # -------------------------

        if (
            early_stopping
            and restore_best_weights
            and best_weights is not None
        ):
            i = 0

            for layer in self.layers:
                if hasattr(layer, "weights"):
                    layer.weights = copy.deepcopy(best_weights[i]["weights"])
                    layer.biases = copy.deepcopy(best_weights[i]["biases"])
                    i += 1

            print(
                f"Restored best weights from epoch {self.best_epoch}."
            )

    def predict(self, X):
        """Performs a forward pass through all layers to generate predictions."""

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def save(self, path):
        """Serializes the model layer parameters into a pickle file."""

        parameters = []

        for layer in self.layers:
            if hasattr(layer, "weights"):
                parameters.append([
                    layer.weights,
                    layer.biases
                ])

        data_to_save = {
            "parameters": parameters
        }

        with open(path, "wb") as file:
            pickle.dump(data_to_save, file)

        print(f"Model saved to {path}")

    def load(self, path):
        """Restores model layer weights and biases from a pickle file."""

        with open(path, "rb") as file:
            data = pickle.load(file)

        i = 0

        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.weights = data["parameters"][i][0]
                layer.biases = data["parameters"][i][1]
                i += 1

        print(f"Model loaded from {path}")