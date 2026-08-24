import numpy as np
import pickle
import pandas as pd

import layers
import activation_functions
import loss_functions
import optimizers
from data.scaler import MinMaxScaler 
from data.dataloader import DataLoader
from metrics.metrics import Accuracy
from data.utils import train_test_split
from model import Model 

# Initialize controlled random number generator for reproducibility
rng = np.random.default_rng(112)

# Load dataset & extract target/features
data = pd.read_csv('IRIS.csv')

class_mapping = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

inverse_mapping = {v: k for k, v in class_mapping.items()}

data['species'] = data['species'].map(inverse_mapping)

y = data['species'].values
X = data[['sepal_length', 
          'sepal_width', 
          'petal_length', 
          'petal_width'
          ]].to_numpy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, rng=rng)

# Fit scalers on train set only to prevent data leakage, then scale features 
model = Model(scaler_X=MinMaxScaler())
model.scaler_X.fit(X_train)
X_train = model.scaler_X.transform(X_train)
X_test = model.scaler_X.transform(X_test)

# Setup batch iterators with deterministic shuffling
train_loader = DataLoader(X_train, y_train, batch_size=32, rng=rng)
test_loader = DataLoader(X_test, y_test, batch_size=32, rng=rng)

# Build architecture
model.add(layers.Layer_Dense(4, 4, activation_func=activation_functions.ReLU(), rng=rng))
model.add(layers.Layer_Dense(4, 3, activation_func=activation_functions.Softmax(), rng=rng))

# Compile with loss and optimizer
model.compile(
    loss=loss_functions.CrossEntropy(),
    optimizer=optimizers.Adam(learning_rate=0.01)
)

print("Training model...")

train_metrics = [Accuracy()]
test_metrics = [Accuracy()]

# Run training loop with validation
model.fit(
    train_dataloader=train_loader, 
    test_dataloader=test_loader, 
    epochs=500, 
    train_metrics=train_metrics, 
    test_metrics=test_metrics
) 