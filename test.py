import layers
import activation_functions
import loss_functions
import numpy as np

inputs = [1, 2, 3, 4, 5]
inputs = np.array(inputs).reshape(-1, 1)

y_true = [4, 8, 12, 16, 20]
y_true = np.array(y_true).reshape(-1, 1)

layer1 = layers.Layer_Dense(1, 4)
layer1.forward(inputs)

activation1 = activation_functions.Activation_ReLU()
activation1.forward(layer1.output)

layer2 = layers.Layer_Dense(4, 1)
layer2.forward(activation1.output)

activation2 = activation_functions.Activation_Linear()
guess = activation2.forward(layer2.output)

print(guess)

mse = loss_functions.mse()
error = mse.forward(guess, y_true)

print(error)