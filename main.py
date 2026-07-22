import numpy as np
import pickle
import pandas as pd

import layers
import activation_functions
import loss_functions
import optimizers
from data.scaler import MinMaxScaler 
from data.dataloader import DataLoader
from metrics.metrics import MAPE
from data.utils import train_test_split
from model import Model 

# Load dataset & extract target/features
data = pd.read_csv('housing.csv')
y = data['median_house_value'].values.reshape(-1, 1)
X = data['total_rooms'].values.reshape(-1, 1)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit scaler on train only to prevent data leakage, then transform both
model = Model(scaler=MinMaxScaler())
model.scaler.fit(X_train)
X_train = model.scaler.transform(X_train)
X_test = model.scaler.transform(X_test)

# Setup batch iterators
train_loader = DataLoader(X_train, y_train, batch_size=32)
test_loader = DataLoader(X_test, y_test, batch_size=32)

# Build architecture
model.add(layers.Layer_Dense(1, 4, activation_func=activation_functions.ReLU()))
model.add(layers.Layer_Dense(4, 4, activation_func=activation_functions.ReLU()))
model.add(layers.Layer_Dense(4, 1, activation_func=activation_functions.Linear()))

# Compile with loss and optimizer
model.compile(
    loss=loss_functions.Mse(),
    optimizer=optimizers.SGD(learning_rate=0.0001)
)

print("Training model...")

train_metrics = [MAPE()]
test_metrics = [MAPE()]

# Run training loop with validation
model.fit(
    train_dataloader=train_loader, 
    test_dataloader=test_loader, 
    epochs=500, 
    train_metrics=train_metrics, 
    test_metrics=test_metrics
) 

print("\nSaving model weights...")
model.save('house_pricing_weights.pkl')