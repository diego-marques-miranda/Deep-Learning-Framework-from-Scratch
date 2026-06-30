# 🧠 Deep Learning Framework from Scratch

A lightweight, educational Deep Learning framework built entirely from scratch using only Python and NumPy.

No TensorFlow. No PyTorch. No Keras. Just pure linear algebra, calculus, and software engineering.

🎯 The Motivation

The goal of this project was to break the "black box" of modern machine learning libraries. By implementing the forward and backward passes from scratch, this framework demonstrates a deep understanding of:

Chain Rule & Backpropagation

Gradient Descent optimization

The vanishing/exploding gradient problem (and how to fix it with weight initialization and learning rate tuning)

The "Dying ReLU" phenomenon

⚙️ Features

Core Architecture: Object-Oriented Model class with dynamic layer stacking (add), compile, fit, and predict methods.

Layers: Layer_Dense (Fully Connected) with He Initialization.

Activations: ReLU (Rectified Linear Unit) and Linear (Identity).

Loss Functions: MSE (Mean Squared Error).

Optimizers: SGD (Stochastic Gradient Descent).

Metrics: Real-time Train/Validation MAPE (Mean Absolute Percentage Error) calculation.

Serialization: Save and load trained weights using binary .pkl files.

🚀 Quick Start (Example: House Pricing Regression)

import numpy as np
import layers, activation_functions, loss_functions, optimizers
from model import Model 

# 1. Prepare Data (X_train, y_train must be scaled!)
# ...

# 2. Build the Network Architecture
model = Model()
model.add(layers.Layer_Dense(2, 8, activation_func=activation_functions.Activation_ReLU()))
model.add(layers.Layer_Dense(8, 4, activation_func=activation_functions.Activation_ReLU()))
model.add(layers.Layer_Dense(4, 1, activation_func=activation_functions.Activation_Linear()))

# 3. Compile
model.compile(
    loss=loss_functions.mse(),
    optimizer=optimizers.SGD(learning_rate=0.01)
)

# 4. Train
model.fit(X_train, y_train, epochs=600, batch_size=16, validation_split=0.2) 

# 5. Save & Predict
model.save('model_weights.pkl')
predictions = model.predict(X_new)


🛠️ Upcoming Features (Roadmap)

[ ] Adam Optimizer

[ ] Softmax Activation & Cross-Entropy Loss for Classification

[ ] Dropout Layer for Regularization

[ ] Convolutional Layers (CNNs)

🤝 Contributing

This is an educational project, but suggestions, optimizations, and pull requests are always welcome!