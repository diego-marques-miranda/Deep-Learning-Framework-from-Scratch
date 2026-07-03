import numpy as np
import pickle

import layers
import activation_functions
import loss_functions
import optimizers
from model import Model 

# Generate synthetic dataset for house pricing regression
# Target formula: Price = (3.5 * bedrooms) + (1.2 * age) + 50
np.random.seed(42)
samples = 200

X_bedrooms = np.random.randint(1, 6, size=(samples, 1))
X_age = np.random.randint(0, 51, size=(samples, 1))
X = np.hstack((X_bedrooms, X_age))

# Target variable with added Gaussian noise
noise = np.random.randn(samples, 1) * 2
y_true = (3.5 * X_bedrooms) + (1.2 * X_age) + 50 + noise

# Min-Max scaling to prevent gradient explosion and dying ReLUs
X_scaled = X / np.array([5.0, 50.0])

# Model architecture definition
model = Model()

# Hidden layers
model.add(layers.Layer_Dense(2, 8, activation_func=activation_functions.ReLU()))
model.add(layers.Layer_Dense(8, 4, activation_func=activation_functions.ReLU()))

# Output layer (linear activation for regression)
model.add(layers.Layer_Dense(4, 1, activation_func=activation_functions.Activation_Linear()))

# Configure loss function and optimizer
model.compile(
    loss=loss_functions.mse(),
    optimizer=optimizers.SGD(learning_rate=0.0001)
)

print("Training model...")
model.fit(X_scaled, y_true, epochs=600, batch_size=16, validation_split=0.2) 

print("\nSaving model weights...")
model.save('house_pricing_weights.pkl')

# Instantiate a new model to validate weight loading
print("\nInitializing a fresh model instance...")
prod_model = Model()
prod_model.add(layers.Layer_Dense(2, 8, activation_func=activation_functions.ReLU()))
prod_model.add(layers.Layer_Dense(8, 4, activation_func=activation_functions.ReLU()))
prod_model.add(layers.Layer_Dense(4, 1, activation_func=activation_functions.Linear()))

# Load pre-trained weights
prod_model.load('house_pricing_weights.pkl')

print("\nRunning inference on new data:")
# Sample data: [bedrooms, age]
# Expected approximate targets: ~72.5, ~67.5, ~113.5
X_new = np.array([
    [3, 10],
    [5, 0],
    [1, 50]
])

# Apply the same scaling parameters used during training
X_new_scaled = X_new / np.array([5.0, 50.0])

predictions = prod_model.predict(X_new_scaled)

for i, house in enumerate(X_new):
    print(f"House [{house[0]} bedrooms, {house[1]} years old] -> Predicted Price: ${predictions[i][0]:.2f}k")